"""Fast, offline verification for :mod:`core`.

This is deliberately a plain Python script rather than a test-framework test so
the DPR core can be checked in a fresh environment containing only PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

if __package__:
    from .core import (
        DPRBiEncoder,
        answer_containment_at_k,
        contains_answer,
        exact_topk_accuracy,
        in_batch_nll_loss,
    )
else:
    from core import (
        DPRBiEncoder,
        answer_containment_at_k,
        contains_answer,
        exact_topk_accuracy,
        in_batch_nll_loss,
    )


@dataclass
class TinyEncoderOutput:
    """The one output field consumed by ``DPRBiEncoder``."""

    last_hidden_state: Tensor


class TinyRandomEncoder(nn.Module):
    """A tiny contextual encoder used to exercise first-token pooling."""

    def __init__(self, vocabulary_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size, padding_idx=0)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> TinyEncoderOutput:
        del token_type_ids  # Accepted to mirror a BERT-like encoder signature.
        token_vectors = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        mask = attention_mask.to(token_vectors.dtype).unsqueeze(-1)
        context = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        cls_vector = self.layer_norm(
            token_vectors[:, 0, :] + self.context_projection(context)
        )
        # Position zero is contextual, while retaining a normal sequence output.
        sequence = torch.cat((cls_vector[:, None, :], token_vectors[:, 1:, :]), dim=1)
        return TinyEncoderOutput(last_hidden_state=sequence)


def _toy_inputs() -> tuple[Tensor, Tensor, Tensor, list[str], list[list[str]]]:
    # Every row starts with token 1, which plays the role of [CLS].
    questions = torch.tensor(
        [
            [1, 2, 10, 0],
            [1, 3, 11, 0],
            [1, 4, 12, 0],
            [1, 5, 13, 0],
        ]
    )
    positives = torch.tensor(
        [
            [1, 2, 20, 0],
            [1, 3, 21, 0],
            [1, 4, 22, 0],
            [1, 5, 23, 0],
        ]
    )
    hard_negatives = torch.tensor(
        [
            [1, 3, 20, 0],
            [1, 4, 21, 0],
            [1, 5, 22, 0],
            [1, 2, 23, 0],
        ]
    )
    # Text order follows core.interleave_passages: positive then hard negative.
    passage_texts = [
        "Marie Curie discovered radium.",
        "This biography discusses a different scientist.",
        "The capital of Japan is Tokyo.",
        "Kyoto was an earlier imperial capital.",
        "The Nile is a major river in Africa.",
        "The Niger is another African river.",
        "Python was created by Guido van Rossum.",
        "Monty Python was a comedy group.",
    ]
    answers = [["radium"], ["Tokyo"], ["the Nile", "Nile"], ["Guido van Rossum"]]
    return questions, positives, hard_negatives, passage_texts, answers


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)

    question_encoder = TinyRandomEncoder()
    passage_encoder = TinyRandomEncoder()
    model = DPRBiEncoder(question_encoder, passage_encoder)
    assert model.question_encoder is not model.passage_encoder

    questions, positives, hard_negatives, passage_texts, answers = _toy_inputs()
    question_mask = questions.ne(0)
    positive_mask = positives.ne(0)
    hard_negative_mask = hard_negatives.ne(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    def compute_loss() -> tuple[Tensor, Tensor, Tensor]:
        question_vectors = model.encode_questions(questions, question_mask)
        positive_vectors = model.encode_passages(positives, positive_mask)
        hard_negative_vectors = model.encode_passages(
            hard_negatives, hard_negative_mask
        )
        return in_batch_nll_loss(
            question_vectors, positive_vectors, hard_negative_vectors
        )

    model.train()
    initial_loss, _, _ = compute_loss()
    initial_value = initial_loss.item()

    # The first update explicitly verifies that both independent towers receive
    # gradients and that optimization changes their parameters.
    question_before = question_encoder.embedding.weight.detach().clone()
    passage_before = passage_encoder.embedding.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    initial_loss.backward()
    question_grad = question_encoder.embedding.weight.grad
    passage_grad = passage_encoder.embedding.weight.grad
    assert question_grad is not None and question_grad.norm().item() > 0.0
    assert passage_grad is not None and passage_grad.norm().item() > 0.0
    optimizer.step()
    assert not torch.equal(question_before, question_encoder.embedding.weight)
    assert not torch.equal(passage_before, passage_encoder.embedding.weight)

    for _ in range(119):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_loss()
        loss.backward()
        optimizer.step()

    model.eval()
    final_loss, scores, labels = compute_loss()
    final_value = final_loss.item()
    assert final_value < initial_value * 0.10, (initial_value, final_value)
    assert labels.tolist() == [0, 2, 4, 6]
    assert scores.shape == (4, 8)
    assert torch.equal(scores.argmax(dim=1), labels), (scores, labels)

    exact = exact_topk_accuracy(scores, labels, ks=(1, 2))
    containment = answer_containment_at_k(
        scores, passage_texts, answers, ks=(1, 2)
    )
    assert exact == {1: 1.0, 2: 1.0}
    assert containment == {1: 1.0, 2: 1.0}
    assert contains_answer("PARIS is in France.", ["Paris"])
    assert not contains_answer("A Parisian cafe.", ["Paris"])

    # A non-perfect fixed ranking makes sure top-k is cumulative and exact.
    metric_scores = torch.tensor([[9.0, 8.0, 1.0], [1.0, 3.0, 2.0]])
    metric_exact = exact_topk_accuracy(metric_scores, [0, 2], ks=(1, 2))
    metric_answers = answer_containment_at_k(
        metric_scores,
        ["alpha is here", "beta is here", "unrelated"],
        [["alpha"], ["beta"]],
        ks=(1, 2),
    )
    assert metric_exact == {1: 0.5, 2: 1.0}
    assert metric_answers == {1: 1.0, 2: 1.0}

    print("DPR core verification passed")
    print(f"loss: {initial_value:.4f} -> {final_value:.4f}")
    print(f"exact@1={exact[1]:.3f}, answer_containment@1={containment[1]:.3f}")


if __name__ == "__main__":
    main()
