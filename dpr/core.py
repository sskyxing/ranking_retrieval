"""Minimal, PyTorch-only core of Dense Passage Retrieval (DPR).

The module intentionally leaves tokenisation and the encoder architecture to the
caller.  An encoder only needs to accept Hugging Face-style tensor arguments and
return a rank-3 last hidden state.  The first token (normally ``[CLS]``) is used
as the dense representation, matching the original DPR formulation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _last_hidden_state(encoder_output: Any) -> Tensor:
    """Extract a sequence tensor from common PyTorch encoder return types."""

    if isinstance(encoder_output, Tensor):
        hidden = encoder_output
    elif hasattr(encoder_output, "last_hidden_state"):
        hidden = encoder_output.last_hidden_state
    elif isinstance(encoder_output, Mapping) and "last_hidden_state" in encoder_output:
        hidden = encoder_output["last_hidden_state"]
    elif isinstance(encoder_output, (tuple, list)) and encoder_output:
        hidden = encoder_output[0]
    else:
        raise TypeError(
            "encoder must return a Tensor or an object containing "
            "'last_hidden_state'"
        )

    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise ValueError(
            "last_hidden_state must be a Tensor with shape "
            "[batch, sequence_length, hidden_size]"
        )
    if hidden.shape[1] == 0:
        raise ValueError("the encoder returned an empty token sequence")
    return hidden


def cls_representation(encoder_output: Any) -> Tensor:
    """Return the first-token representation with shape ``[batch, hidden]``."""

    return _last_hidden_state(encoder_output)[:, 0, :]


class DPRBiEncoder(nn.Module):
    """DPR's two independent question and passage encoders.

    The encoder modules must be distinct objects.  They may start from identical
    weights, but DPR updates their parameters independently.
    """

    def __init__(self, question_encoder: nn.Module, passage_encoder: nn.Module) -> None:
        super().__init__()
        if question_encoder is passage_encoder:
            raise ValueError("question_encoder and passage_encoder must be independent")
        question_parameter_ids = {id(parameter) for parameter in question_encoder.parameters()}
        if any(
            id(parameter) in question_parameter_ids
            for parameter in passage_encoder.parameters()
        ):
            raise ValueError(
                "question_encoder and passage_encoder must not share parameters"
            )
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder

    @staticmethod
    def _encode(
        encoder: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length]")
        kwargs: dict[str, Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            if token_type_ids.shape != input_ids.shape:
                raise ValueError("token_type_ids must have the same shape as input_ids")
            kwargs["token_type_ids"] = token_type_ids
        return cls_representation(encoder(**kwargs))

    def encode_questions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """Encode questions using the question tower's first token."""

        return self._encode(
            self.question_encoder, input_ids, attention_mask, token_type_ids
        )

    def encode_passages(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """Encode passages using the passage tower's first token."""

        return self._encode(
            self.passage_encoder, input_ids, attention_mask, token_type_ids
        )

    def forward(
        self,
        question_input_ids: Tensor,
        passage_input_ids: Tensor,
        question_attention_mask: Tensor | None = None,
        passage_attention_mask: Tensor | None = None,
        question_token_type_ids: Tensor | None = None,
        passage_token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return question and passage vectors; scoring is kept explicit."""

        question_vectors = self.encode_questions(
            question_input_ids,
            question_attention_mask,
            question_token_type_ids,
        )
        passage_vectors = self.encode_passages(
            passage_input_ids,
            passage_attention_mask,
            passage_token_type_ids,
        )
        return question_vectors, passage_vectors


def dot_product_scores(question_vectors: Tensor, passage_vectors: Tensor) -> Tensor:
    """Compute every question-passage inner product."""

    if question_vectors.ndim != 2 or passage_vectors.ndim != 2:
        raise ValueError("question_vectors and passage_vectors must both be rank 2")
    if question_vectors.shape[1] != passage_vectors.shape[1]:
        raise ValueError("question and passage hidden sizes must match")
    return question_vectors @ passage_vectors.transpose(0, 1)


def interleave_passages(
    positive_vectors: Tensor, hard_negative_vectors: Tensor
) -> Tensor:
    """Lay passages out as ``[p0+, p0-, p1+, p1-, ...]``."""

    if positive_vectors.ndim != 2 or hard_negative_vectors.ndim != 2:
        raise ValueError("passage vectors must be rank 2")
    if positive_vectors.shape != hard_negative_vectors.shape:
        raise ValueError(
            "each question must have exactly one positive and one hard negative"
        )
    return torch.stack((positive_vectors, hard_negative_vectors), dim=1).flatten(0, 1)


def in_batch_nll_loss(
    question_vectors: Tensor,
    positive_vectors: Tensor,
    hard_negative_vectors: Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute DPR's all-in-batch softmax negative log-likelihood.

    For a batch of size ``B``, the candidate matrix contains ``2B`` passages:
    one positive and one hard negative from every question.  Thus, besides its
    paired hard negative, each question also sees every other passage in the
    batch as a negative.

    Returns:
        ``(loss, scores, positive_indices)`` where scores has shape ``[B, 2B]``.
    """

    if question_vectors.ndim != 2:
        raise ValueError("question_vectors must be rank 2")
    if question_vectors.shape != positive_vectors.shape:
        raise ValueError(
            "question_vectors and positive_vectors must have the same shape"
        )
    if question_vectors.shape[0] == 0:
        raise ValueError("an empty batch has no DPR loss")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")

    candidates = interleave_passages(positive_vectors, hard_negative_vectors)
    scores = dot_product_scores(question_vectors, candidates) / temperature
    positive_indices = torch.arange(
        question_vectors.shape[0], device=scores.device, dtype=torch.long
    ) * 2
    loss = F.cross_entropy(scores, positive_indices)
    return loss, scores, positive_indices


def _positive_index_tensor(
    positive_indices: Tensor | Sequence[int], scores: Tensor
) -> Tensor:
    indices = torch.as_tensor(positive_indices, dtype=torch.long, device=scores.device)
    if indices.ndim != 1 or indices.shape[0] != scores.shape[0]:
        raise ValueError(
            "positive_indices must contain one index for every question row"
        )
    if bool(((indices < 0) | (indices >= scores.shape[1])).any()):
        raise ValueError("positive_indices contains an out-of-range passage index")
    return indices


def _validated_ks(ks: Sequence[int], candidate_count: int) -> tuple[int, ...]:
    if candidate_count < 1:
        raise ValueError("at least one passage candidate is required")
    values = tuple(dict.fromkeys(int(k) for k in ks))
    if not values or any(k < 1 for k in values):
        raise ValueError("ks must contain positive integers")
    return values


@torch.no_grad()
def exact_topk_accuracy(
    scores: Tensor,
    positive_indices: Tensor | Sequence[int],
    ks: Sequence[int] = (1, 5, 20, 100),
) -> dict[int, float]:
    """Fraction of questions whose exact gold passage index is in top ``k``."""

    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("scores must have shape [nonempty questions, passages]")
    requested_ks = _validated_ks(ks, scores.shape[1])
    labels = _positive_index_tensor(positive_indices, scores)
    width = min(max(requested_ks), scores.shape[1])
    ranked = scores.topk(width, dim=1, largest=True, sorted=True).indices
    return {
        k: float((ranked[:, : min(k, width)] == labels[:, None]).any(dim=1).float().mean())
        for k in requested_ks
    }


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _answer_tokens(text: str) -> list[str]:
    # NFD makes canonically equivalent Unicode spellings compare identically.
    normalized = unicodedata.normalize("NFD", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def contains_answer(passage_text: str, answers: Sequence[str]) -> bool:
    """Test exact, case-insensitive token-sequence answer containment.

    Token matching prevents an answer such as ``"Paris"`` from matching
    ``"Parisian"``.  Empty answer strings are ignored.
    """

    passage_tokens = _answer_tokens(passage_text)
    for answer in answers:
        answer_tokens = _answer_tokens(answer)
        if not answer_tokens or len(answer_tokens) > len(passage_tokens):
            continue
        window_count = len(passage_tokens) - len(answer_tokens) + 1
        if any(
            passage_tokens[start : start + len(answer_tokens)] == answer_tokens
            for start in range(window_count)
        ):
            return True
    return False


@torch.no_grad()
def answer_containment_at_k(
    scores: Tensor,
    passage_texts: Sequence[str],
    answers_per_question: Sequence[Sequence[str]],
    ks: Sequence[int] = (1, 5, 20, 100),
) -> dict[int, float]:
    """Fraction of questions with an answer string in any top-``k`` passage."""

    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("scores must have shape [nonempty questions, passages]")
    if len(passage_texts) != scores.shape[1]:
        raise ValueError("passage_texts must align with the columns of scores")
    if len(answers_per_question) != scores.shape[0]:
        raise ValueError("answers_per_question must align with the rows of scores")

    requested_ks = _validated_ks(ks, scores.shape[1])
    width = min(max(requested_ks), scores.shape[1])
    ranked = scores.topk(width, dim=1, largest=True, sorted=True).indices.cpu()
    hit_totals = {k: 0 for k in requested_ks}

    for question_index, answer_variants in enumerate(answers_per_question):
        ranked_indices = ranked[question_index].tolist()
        for k in requested_ks:
            if any(
                contains_answer(passage_texts[index], answer_variants)
                for index in ranked_indices[: min(k, width)]
            ):
                hit_totals[k] += 1

    question_count = scores.shape[0]
    return {k: hit_totals[k] / question_count for k in requested_ks}


__all__ = [
    "DPRBiEncoder",
    "answer_containment_at_k",
    "cls_representation",
    "contains_answer",
    "dot_product_scores",
    "exact_topk_accuracy",
    "in_batch_nll_loss",
    "interleave_passages",
]
