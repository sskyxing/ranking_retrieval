"""Document-aware NFCorpus fine-tuning and full-corpus dev evaluation for DPR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from smile_lightning.apps.retrieval.dpr.model import DPRBiEncoder


def load_nq_tower_weights(model: DPRBiEncoder, checkpoint_path: str) -> tuple[int, int]:
    """Warm-start encoders only; never restore NQ optimizer, scheduler or epoch."""

    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = sorted(path.rglob("*.ckpt"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one mounted NQ checkpoint, found {len(candidates)} in {path}.")
        path = candidates[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    for tower in ("question", "context"):
        prefix = f"model.{tower}_encoder."
        weights = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if not weights:
            raise ValueError(f"NQ checkpoint lacks {tower} encoder weights.")
        getattr(model, f"{tower}_encoder").load_state_dict(weights, strict=True)
    return int(checkpoint["epoch"]), int(checkpoint["global_step"])


def gather_variable_rows(local: Tensor) -> Tensor:
    """Gather uneven rank shards while preserving autograd for this rank."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    world_size, rank = dist.get_world_size(), dist.get_rank()
    local_length = torch.tensor([local.shape[0]], device=local.device, dtype=torch.long)
    lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
    dist.all_gather(lengths, local_length)
    sizes = [int(length.item()) for length in lengths]
    padded = local.detach().contiguous()
    if local.shape[0] < max(sizes):
        padded = torch.cat((padded, padded.new_zeros((max(sizes) - local.shape[0], *local.shape[1:]))), dim=0)
    shards = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(shards, padded)
    trimmed = [shard[:size] for shard, size in zip(shards, sizes)]
    trimmed[rank] = local
    return torch.cat(trimmed, dim=0)


def max_document_scores(query_vectors: Tensor, chunk_vectors: Tensor, doc_indices: Tensor, doc_count: int):
    """One document score is the largest dot product over its chunks."""

    chunk_scores = query_vectors @ chunk_vectors.T
    scores = chunk_scores.new_full((len(query_vectors), doc_count), -torch.inf)
    columns = doc_indices.long().unsqueeze(0).expand(len(query_vectors), -1)
    return scores.scatter_reduce(1, columns, chunk_scores, reduce="amax", include_self=True)


def similarity_vectors(vectors: Tensor, similarity_mode: str) -> Tensor:
    """Prepare encoder outputs for the selected train and retrieval score."""

    if similarity_mode == "dot":
        return vectors
    if similarity_mode == "cosine":
        return F.normalize(vectors, p=2, dim=-1)
    raise ValueError(f"Unsupported NFCorpus similarity_mode: {similarity_mode}")


def masked_positive_loss(
    scores: Tensor,
    doc_indices: Tensor,
    targets: Tensor,
    known_positive_mask: Tensor,
    weights: Tensor,
    score_scale: float = 1.0,
) -> Tensor:
    """One target per row, excluding every other known positive from negatives."""

    if scores.shape != known_positive_mask.shape or scores.shape[1] != len(doc_indices) or len(targets) != len(scores):
        raise ValueError("NFCorpus score/ID/target dimensions disagree.")
    allowed = ~known_positive_mask.clone()
    target_columns = doc_indices.unsqueeze(0).eq(targets.unsqueeze(1))
    if not target_columns.any(dim=1).all():
        raise ValueError("Target document is absent from the candidate pool.")
    if target_columns.sum(dim=1).max() != 1:
        raise ValueError("Candidate document IDs must be unique.")
    allowed |= target_columns
    logits = (scores * score_scale).masked_fill(~allowed, -torch.inf)
    labels = target_columns.int().argmax(dim=1)
    losses = F.cross_entropy(logits, labels, reduction="none")
    if not torch.isfinite(losses).all():
        raise ValueError("NFCorpus training loss is not finite.")
    return (losses * weights).mean()


def macro_retrieval_metrics(scores: Tensor, query_ids: List[str], doc_ids: List[str], relevance):
    """Match bm25.run_bm25.evaluate: macro Recall@10 and linear-gain nDCG@10."""

    recalls, ndcgs = [], []
    skipped = []
    for index, query_id in enumerate(query_ids):
        positives = relevance.get(query_id, {})
        if not positives:
            skipped.append(query_id)
            continue
        top_indices = torch.argsort(scores[index], descending=True, stable=True)[:10].tolist()
        retrieved = [doc_ids[column] for column in top_indices]
        recalls.append(sum(doc_id in positives for doc_id in retrieved) / len(positives))
        dcg = sum(positives.get(doc_id, 0) / math.log2(rank + 1) for rank, doc_id in enumerate(retrieved, 1))
        ideal = sorted(positives.values(), reverse=True)[:10]
        idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        ndcgs.append(dcg / idcg)
    if not recalls:
        raise ValueError("Dev split has no queries with positive documents.")
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs), skipped


def broadcast_rank_zero_metrics(
    recall: Optional[float], ndcg: Optional[float], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return rank-zero retrieval metrics on every distributed rank.

    Callback monitors must exist with the same value on every rank. Rank zero
    computes the full-corpus metrics while the remaining ranks receive them via
    a broadcast before all ranks call ``self.log``.
    """

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        if recall is None or ndcg is None:
            raise ValueError("Rank zero must provide NFCorpus retrieval metrics.")
        metrics = torch.tensor([recall, ndcg], dtype=torch.float64, device=device)
    else:
        metrics = torch.zeros(2, dtype=torch.float64, device=device)
    if distributed:
        dist.broadcast(metrics, src=0)
    return metrics[0], metrics[1]


class NFCorpusLightningModule(LightningModule):
    """Fine-tune NQ DPR towers on NFCorpus, selecting full dev Recall@10."""

    def __init__(
        self,
        model: DPRBiEncoder,
        nq_checkpoint_path: str,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 10,
        max_steps: Optional[int] = None,
        score_scale: float = 1.0,
        similarity_mode: str = "dot",
    ):
        super().__init__()
        if not nq_checkpoint_path:
            raise ValueError("An NQ DPR checkpoint is required for NFCorpus adaptation.")
        if learning_rate <= 0 or warmup_steps < 0 or score_scale <= 0:
            raise ValueError("Invalid NFCorpus optimizer/score settings.")
        if similarity_mode not in ("dot", "cosine"):
            raise ValueError("similarity_mode must be 'dot' or 'cosine'.")
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.nq_epoch, self.nq_step = load_nq_tower_weights(model, nq_checkpoint_path)
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.score_scale = score_scale
        self.similarity_mode = similarity_mode
        self._dev_context_vectors = None
        self._dev_context_doc_indices = None
        self._dev_queries = {}

    def on_train_epoch_start(self):
        dataset = self.trainer.datamodule.train_dataset
        dataset.set_epoch(self.current_epoch)

    def training_step(self, batch: Dict, batch_idx: int):
        del batch_idx
        q = similarity_vectors(self.model.encode_questions(**batch["question"]), self.similarity_mode)
        chunks = similarity_vectors(self.model.encode_contexts(**batch["context"]), self.similarity_mode)
        chunk_doc_indices = batch["chunk_global_doc_indices"].to(chunks.device)
        qids = batch["query_indices"].to(chunks.device)
        targets = batch["target_doc_indices"].to(chunks.device)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            q = gather_variable_rows(q)
            chunks = gather_variable_rows(chunks)
            chunk_doc_indices = gather_variable_rows(chunk_doc_indices)
            qids = gather_variable_rows(qids)
            targets = gather_variable_rows(targets)
        else:
            world_size = 1
        # Multiple ranks may carry the same document. Aggregate all its chunks
        # into one column before applying document-aware positive masking.
        doc_indices, inverse = torch.unique(chunk_doc_indices, sorted=True, return_inverse=True)
        scores = max_document_scores(q, chunks, inverse, len(doc_indices))
        data = self.trainer.datamodule
        candidate_ids = [data.train_doc_ids[int(index)] for index in doc_indices.tolist()]
        query_ids = [data.train_query_ids[int(index)] for index in qids.tolist()]
        known_mask = torch.tensor(
            [[did in data.train_positive_sets[qid] for did in candidate_ids] for qid in query_ids],
            dtype=torch.bool,
            device=scores.device,
        )
        weight_scale = len(data.train_dataset) / len(data.train_query_ids)
        weights = torch.tensor(
            [weight_scale / data.train_positive_counts[qid] for qid in query_ids],
            dtype=scores.dtype,
            device=scores.device,
        )
        loss = masked_positive_loss(scores, doc_indices, targets, known_mask, weights, self.score_scale)
        self.log("train/nll", loss, on_step=True, on_epoch=True, batch_size=len(qids), sync_dist=world_size > 1)
        # DDP averages parameter gradients, whereas each rank retains only
        # local embedding gradients from the globally computed loss.
        return loss * world_size

    def on_validation_epoch_start(self):
        # Rank zero encodes the complete dev corpus in bounded batches; all
        # ranks contribute their query vectors to one full-corpus metric.
        self._dev_queries = {}
        if not self.trainer.is_global_zero:
            return
        datamodule = self.trainer.datamodule
        doc_vectors, doc_indices = [], []
        with torch.inference_mode():
            for features, indices in datamodule.dev_document_batches(datamodule.val_batch_size):
                features = {key: value.to(self.device) for key, value in features.items()}
                encoded = self.model.encode_contexts(**features)
                doc_vectors.append(similarity_vectors(encoded, self.similarity_mode).float().cpu())
                doc_indices.extend(indices)
        self._dev_context_vectors = torch.cat(doc_vectors)
        self._dev_context_doc_indices = torch.tensor(doc_indices, dtype=torch.long)

    def validation_step(self, batch: Dict, batch_idx: int):
        del batch_idx
        features = {key: value.to(self.device) for key, value in batch["question"].items()}
        with torch.inference_mode():
            encoded = self.model.encode_questions(**features)
            vectors = similarity_vectors(encoded, self.similarity_mode).float().cpu()
        for qid, vector in zip(batch["query_ids"], vectors):
            self._dev_queries[qid] = vector

    def on_validation_epoch_end(self):
        datamodule = self.trainer.datamodule
        hidden_size = self.model.question_encoder.config.hidden_size
        query_sums = torch.zeros((len(datamodule.dev_query_ids), hidden_size), device=self.device)
        query_counts = torch.zeros(len(datamodule.dev_query_ids), device=self.device)
        for qid, vector in self._dev_queries.items():
            index = datamodule.dev_query_id_to_index[qid]
            query_sums[index] = vector.to(self.device)
            query_counts[index] = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(query_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(query_counts, op=dist.ReduceOp.SUM)
        present = bool((query_counts > 0).all().item())
        if not present:
            raise RuntimeError("Distributed validation omitted NFCorpus dev queries.")
        if self.trainer.is_global_zero and len(self._dev_context_vectors) != len(self._dev_context_doc_indices):
            raise RuntimeError("NFCorpus dev passage embeddings and document IDs differ in length.")
        recall = ndcg = None
        if self.trainer.is_global_zero:
            query_vectors = (query_sums / query_counts.unsqueeze(1)).cpu()
            with torch.inference_mode():
                scores = max_document_scores(
                    query_vectors,
                    self._dev_context_vectors,
                    self._dev_context_doc_indices,
                    len(datamodule.dev_doc_ids),
                )
                recall, ndcg, _ = macro_retrieval_metrics(
                    scores.float().cpu(), datamodule.dev_query_ids, datamodule.dev_doc_ids, datamodule.dev_relevance
                )
        recall, ndcg = broadcast_rank_zero_metrics(recall, ndcg, self.device)
        # Every rank must expose identical callback metrics. In particular, a
        # rank-zero-only metric cannot be used by Lightning's checkpoint monitor.
        self.log("val/recall_at_10", recall, on_step=False, on_epoch=True)
        self.log("val/ndcg_at_10", ndcg, on_step=False, on_epoch=True)
        self._dev_context_vectors = None
        self._dev_context_doc_indices = None
        self._dev_queries = {}

    def configure_optimizers(self):
        no_decay_names = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight")
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            (no_decay if any(item in name for item in no_decay_names) else decay).append(parameter)
        optimizer = AdamW(
            [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.learning_rate,
            eps=self.adam_eps,
        )
        total_steps = self.max_steps or self.trainer.estimated_stepping_batches
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("NFCorpus trainer must have a finite positive step count.")

        def lr_factor(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - self.warmup_steps))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_factor), "interval": "step"},
        }
