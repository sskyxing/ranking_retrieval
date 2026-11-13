"""Lightning training loop and distributed in-batch negatives for DPR."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import DPRBiEncoder


def validate_equal_batch_sizes(batch_sizes: Sequence[int], tensor_name: str = "tensor") -> int:
    """Reject uneven DDP shards before a fixed-shape all-gather can hang."""

    if not batch_sizes:
        raise ValueError("batch_sizes cannot be empty.")
    normalized = [int(size) for size in batch_sizes]
    if normalized[0] <= 0:
        raise RuntimeError(f"{tensor_name} has an empty local batch.")
    if any(size != normalized[0] for size in normalized[1:]):
        raise RuntimeError(
            f"DPR DDP requires equal local batch sizes for {tensor_name}; got {normalized}. "
            "Use a distributed sampler with padding or drop_last=True."
        )
    return normalized[0]


def concatenate_rank_shards_with_local_grad(gathered: Tensor, local: Tensor, rank: int) -> Tensor:
    """Flatten rank-major shards, replacing this rank with its autograd tensor."""

    if gathered.ndim != local.ndim + 1:
        raise ValueError("Gathered tensor must have one leading rank dimension.")
    if rank < 0 or rank >= gathered.shape[0]:
        raise ValueError(f"Rank {rank} is outside gathered world size {gathered.shape[0]}.")
    if tuple(gathered.shape[1:]) != tuple(local.shape):
        raise ValueError("Gathered and local tensor shapes do not match.")
    shards = list(gathered.unbind(0))
    shards[rank] = local
    return torch.cat(shards, dim=0)


def _offset_rank_major_indices(
    indices: Tensor, question_counts: Sequence[int], context_counts: Sequence[int]
) -> Tensor:
    """Convert rank-local positive columns to offsets in a global passage pool."""

    if (
        len(question_counts) != len(context_counts)
        or indices.numel() != sum(question_counts)
        or any(size <= 0 for size in context_counts)
    ):
        raise ValueError("Rank-local positive indices do not match gathered question counts.")
    adjusted = []
    question_offset = 0
    context_offset = 0
    for question_count, context_count in zip(question_counts, context_counts):
        local_indices = indices[question_offset : question_offset + question_count]
        if local_indices.numel() and (local_indices.min() < 0 or local_indices.max() >= context_count):
            raise ValueError("A rank-local positive index is outside its context pool.")
        adjusted.append(local_indices + context_offset)
        question_offset += question_count
        context_offset += context_count
    return torch.cat(adjusted, dim=0)


def dpr_positive_indices(num_questions: int, device: torch.device) -> Tensor:
    """Return positive columns for interleaved ``[positive, hard negative]`` contexts."""

    return torch.arange(num_questions, device=device, dtype=torch.long) * 2


def dpr_nll(
    q_vectors: Tensor, context_vectors: Tensor, positive_indices: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute raw dot-product cross entropy and accuracy."""

    if q_vectors.ndim != 2 or context_vectors.ndim != 2:
        raise ValueError("DPR embeddings must be rank-2 tensors.")
    if q_vectors.shape[1] != context_vectors.shape[1]:
        raise ValueError("Question and context embedding dimensions must match.")
    logits = torch.matmul(q_vectors, context_vectors.transpose(0, 1))
    if positive_indices is None:
        if context_vectors.shape[0] != 2 * q_vectors.shape[0]:
            raise ValueError("Training expects exactly two interleaved contexts per question.")
        targets = dpr_positive_indices(q_vectors.shape[0], q_vectors.device)
    else:
        targets = positive_indices.to(device=q_vectors.device, dtype=torch.long)
        if targets.ndim != 1 or targets.shape[0] != q_vectors.shape[0]:
            raise ValueError("There must be one positive context index per question.")
        if targets.numel() and (targets.min() < 0 or targets.max() >= context_vectors.shape[0]):
            raise ValueError("A positive context index is outside the context pool.")
    loss = F.cross_entropy(logits, targets)
    accuracy = logits.argmax(dim=1).eq(targets).float().mean()
    return loss, accuracy, logits


def exact_average_rank(
    q_vectors: Tensor,
    context_vectors: Tensor,
    positive_indices: Optional[Tensor] = None,
    context_chunk_size: int = 4096,
    query_chunk_size: int = 256,
) -> Tensor:
    """Compute exact mean positive rank without materializing the full score matrix.

    Embeddings may be on CPU. Peak score memory is bounded by
    ``query_chunk_size * context_chunk_size`` while embeddings remain O(ND).
    """

    if context_chunk_size <= 0 or query_chunk_size <= 0:
        raise ValueError("Average-rank chunk sizes must be positive.")
    if positive_indices is None:
        if context_vectors.shape[0] != q_vectors.shape[0] * 2:
            raise ValueError("Interleaved average rank expects two contexts per question.")
        positive_indices = dpr_positive_indices(q_vectors.shape[0], q_vectors.device)
    else:
        positive_indices = positive_indices.to(device=q_vectors.device, dtype=torch.long)
        if positive_indices.ndim != 1 or positive_indices.shape[0] != q_vectors.shape[0]:
            raise ValueError("There must be one positive context index per question.")
        if positive_indices.numel() and (
            positive_indices.min() < 0 or positive_indices.max() >= context_vectors.shape[0]
        ):
            raise ValueError("A positive context index is outside the context pool.")
    positive_contexts = context_vectors.index_select(0, positive_indices)
    positive_scores = (q_vectors * positive_contexts).sum(dim=1)
    rank_sum = torch.zeros((), dtype=torch.float64, device=q_vectors.device)

    for query_start in range(0, q_vectors.shape[0], query_chunk_size):
        query_end = min(query_start + query_chunk_size, q_vectors.shape[0])
        query_chunk = q_vectors[query_start:query_end]
        thresholds = positive_scores[query_start:query_end].unsqueeze(1)
        higher_counts = torch.zeros(query_end - query_start, dtype=torch.long, device=q_vectors.device)
        for context_start in range(0, context_vectors.shape[0], context_chunk_size):
            context_end = min(context_start + context_chunk_size, context_vectors.shape[0])
            scores = torch.matmul(query_chunk, context_vectors[context_start:context_end].transpose(0, 1))
            higher_counts += scores.gt(thresholds).sum(dim=1)
        # Match the original DPR implementation: the top result has rank 0.
        rank_sum += higher_counts.double().sum()
    return rank_sum / q_vectors.shape[0]


class DPRLightningModule(LightningModule):
    """Train DPR with global in-batch negatives under DDP.

    Average-rank validation mirrors the original distributed behavior: each
    rank scores its questions against its local validation passage pool, then
    rank sums and question counts are reduced for a weighted global mean.
    """

    def __init__(
        self,
        model: DPRBiEncoder,
        learning_rate: float = 2e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 1237,
        max_steps: Optional[int] = None,
        enable_average_rank: bool = False,
        average_rank_start_epoch: int = 30,
        average_rank_chunk_size: int = 4096,
        validation_context_batch_size: int = 128,
    ) -> None:
        super().__init__()
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied.")
        if average_rank_start_epoch < 0 or average_rank_chunk_size <= 0 or validation_context_batch_size <= 0:
            raise ValueError("Average-rank epoch and chunk configuration is invalid.")

        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.enable_average_rank = enable_average_rank
        self.average_rank_start_epoch = average_rank_start_epoch
        self.average_rank_chunk_size = average_rank_chunk_size
        self.validation_context_batch_size = validation_context_batch_size
        self._average_rank_questions: List[Tensor] = []
        self._average_rank_contexts: List[Tensor] = []
        self._average_rank_positive_indices: List[Tensor] = []
        self._average_rank_context_offset = 0

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.model(
            question_input_ids=batch["question_input_ids"],
            question_attention_mask=batch["question_attention_mask"],
            question_token_type_ids=batch["question_token_type_ids"],
            context_input_ids=batch["context_input_ids"],
            context_attention_mask=batch["context_attention_mask"],
            context_token_type_ids=batch["context_token_type_ids"],
        )

    @staticmethod
    def _validate_local_pairing(q_vectors: Tensor, context_vectors: Tensor, batch: Dict[str, Tensor]) -> None:
        supplied = batch.get("positive_indices")
        if supplied is None or supplied.ndim != 1 or supplied.numel() != q_vectors.shape[0]:
            raise RuntimeError("DPR requires one positive context index per question.")
        if context_vectors.shape[0] < q_vectors.shape[0] or (
            supplied.numel() and (supplied.min() < 0 or supplied.max() >= context_vectors.shape[0])
        ):
            raise RuntimeError("DPR positive context indices must be inside the local context pool.")

    @staticmethod
    def _gather_global(local: Tensor, tensor_name: str, allow_uneven: bool = False) -> Tuple[Tensor, List[int]]:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return local, [local.shape[0]]

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_size = torch.tensor([local.shape[0]], dtype=torch.long, device=local.device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [size.item() for size in gathered_sizes]
        if allow_uneven:
            if any(size <= 0 for size in sizes):
                raise RuntimeError(f"{tensor_name} has an empty rank shard: {sizes}.")
        else:
            validate_equal_batch_sizes(sizes, tensor_name)

        detached_local = local.detach().contiguous()
        if allow_uneven and local.shape[0] < max(sizes):
            padding = detached_local.new_zeros((max(sizes) - local.shape[0], *local.shape[1:]))
            detached_local = torch.cat((detached_local, padding), dim=0)
        shards = [torch.empty_like(detached_local) for _ in range(world_size)]
        dist.all_gather(shards, detached_local)
        if allow_uneven:
            shards = [shard[:size] for shard, size in zip(shards, sizes)]
            shards[rank] = local
            return torch.cat(shards, dim=0), sizes
        return concatenate_rank_shards_with_local_grad(torch.stack(shards, dim=0), local, rank), sizes

    @classmethod
    def _gather_positive_indices(cls, local: Tensor, context_counts: Sequence[int]) -> Tensor:
        rank_major_indices, question_counts = cls._gather_global(local, "positive indices")
        return _offset_rank_major_indices(rank_major_indices, question_counts, context_counts)

    def _shared_step(self, batch: Dict[str, Tensor], stage: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if stage == "train":
            q_local, context_local = self(batch)
            self._validate_local_pairing(q_local, context_local, batch)
            q_vectors, _ = self._gather_global(q_local, "question embeddings")
            context_vectors, context_counts = self._gather_global(
                context_local, "context embeddings", allow_uneven=True
            )
            positive_indices = self._gather_positive_indices(
                batch["positive_indices"].to(context_local.device), context_counts
            )
            loss, accuracy, _ = dpr_nll(q_vectors, context_vectors, positive_indices)
        else:
            q_local = self.model.encode_questions(
                batch["question_input_ids"],
                batch["question_attention_mask"],
                batch["question_token_type_ids"],
            )
            collect_average_rank = self.enable_average_rank and self.current_epoch >= self.average_rank_start_epoch
            context_indices = (
                torch.arange(batch["context_input_ids"].shape[0], device=batch["context_input_ids"].device)
                if collect_average_rank
                else batch["nll_context_indices"]
            )
            context_chunks = []
            for start in range(0, context_indices.shape[0], self.validation_context_batch_size):
                end = start + self.validation_context_batch_size
                chunk_indices = context_indices[start:end]
                context_chunks.append(
                    self.model.encode_contexts(
                        batch["context_input_ids"].index_select(0, chunk_indices),
                        batch["context_attention_mask"].index_select(0, chunk_indices),
                        batch["context_token_type_ids"].index_select(0, chunk_indices),
                    )
                )
            context_vectors = torch.cat(context_chunks, dim=0)
            nll_contexts = (
                context_vectors.index_select(0, batch["nll_context_indices"])
                if collect_average_rank
                else context_vectors
            )
            q_global, _ = self._gather_global(q_local, "validation question embeddings")
            nll_contexts_global, context_counts = self._gather_global(
                nll_contexts, "validation context embeddings", allow_uneven=True
            )
            positive_indices = self._gather_positive_indices(
                batch["nll_positive_indices"].to(nll_contexts.device), context_counts
            )
            loss, accuracy, _ = dpr_nll(q_global, nll_contexts_global, positive_indices)
            q_vectors = q_local
        self.log(
            f"{stage}/nll",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=dist.is_available() and dist.is_initialized(),
            batch_size=q_vectors.shape[0],
        )
        self.log(
            f"{stage}/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
            sync_dist=dist.is_available() and dist.is_initialized(),
            batch_size=q_vectors.shape[0],
        )
        return loss, accuracy, q_vectors, context_vectors

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss, _, _, _ = self._shared_step(batch, "train")
        # Every rank evaluates the same global loss, but only its own gathered
        # embeddings retain autograd history. DDP averages the resulting rank
        # gradients, so undo that average to recover the full-batch gradient.
        if dist.is_available() and dist.is_initialized():
            return loss * dist.get_world_size()
        return loss

    def on_validation_epoch_start(self) -> None:
        self._average_rank_questions.clear()
        self._average_rank_contexts.clear()
        self._average_rank_positive_indices.clear()
        self._average_rank_context_offset = 0

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        loss, _, q_global, context_global = self._shared_step(batch, "val")
        if self.enable_average_rank and self.current_epoch >= self.average_rank_start_epoch:
            self._average_rank_questions.append(q_global.detach().float().cpu())
            self._average_rank_contexts.append(context_global.detach().float().cpu())
            self._average_rank_positive_indices.append(
                batch["positive_indices"].detach().long().cpu() + self._average_rank_context_offset
            )
            self._average_rank_context_offset += context_global.shape[0]
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._average_rank_questions:
            return
        questions = torch.cat(self._average_rank_questions, dim=0)
        contexts = torch.cat(self._average_rank_contexts, dim=0)
        positive_indices = torch.cat(self._average_rank_positive_indices, dim=0)
        average_rank = exact_average_rank(
            questions,
            contexts,
            positive_indices=positive_indices,
            context_chunk_size=self.average_rank_chunk_size,
        ).to(device=self.device, dtype=torch.float64)
        rank_sum = average_rank * questions.shape[0]
        question_count = torch.tensor(float(questions.shape[0]), device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(question_count, op=dist.ReduceOp.SUM)
        average_rank = rank_sum / question_count
        self.log(
            "val/average_rank",
            average_rank,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self._average_rank_questions.clear()
        self._average_rank_contexts.clear()
        self._average_rank_positive_indices.clear()
        self._average_rank_context_offset = 0

    def _total_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            raise RuntimeError("Set max_steps when configure_optimizers is called without a Trainer.")
        estimated = trainer.estimated_stepping_batches
        if not isinstance(estimated, int) or estimated <= 0 or not math.isfinite(estimated):
            raise RuntimeError("Trainer could not determine a finite positive number of optimizer steps.")
        return estimated

    @staticmethod
    def _linear_warmup_decay(step: int, warmup_steps: int, total_steps: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    def configure_optimizers(self):
        no_decay_names = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight")
        decay_parameters: List[nn.Parameter] = []
        no_decay_parameters: List[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay_parameters if any(marker in name for marker in no_decay_names) else decay_parameters
            target.append(parameter)

        optimizer = AdamW(
            [
                {"params": decay_parameters, "weight_decay": self.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            eps=self.adam_eps,
        )
        total_steps = self._total_steps()
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._linear_warmup_decay(step, self.warmup_steps, total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
