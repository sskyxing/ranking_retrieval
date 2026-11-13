from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from smile_lightning.apps.retrieval.dpr.model import DPRBiEncoder
from smile_lightning.apps.retrieval.nfcorpus.module import (
    NFCorpusLightningModule,
    broadcast_rank_zero_metrics,
    load_nq_tower_weights,
    macro_retrieval_metrics,
    masked_positive_loss,
    max_document_scores,
    similarity_vectors,
)


class TinyEncoder(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))
        self.config = SimpleNamespace(hidden_size=1)

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        hidden = input_ids.float().unsqueeze(-1) * self.weight
        return SimpleNamespace(last_hidden_state=hidden)


class StaticBiEncoder(nn.Module):
    """Small encoder with a dot/cosine ranking reversal."""

    def __init__(self):
        super().__init__()
        self.question = nn.Parameter(torch.tensor([[1.0, 0.0]]))
        self.contexts = nn.Parameter(torch.tensor([[4.0, 3.0], [3.0, 0.0]]))
        self.question_encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=2))

    def encode_questions(self, **kwargs):
        return self.question

    def encode_contexts(self, **kwargs):
        return self.contexts


def test_load_nq_tower_weights_restores_only_both_encoders(tmp_path):
    checkpoint_path = tmp_path / "nq.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.question_encoder.weight": torch.tensor([2.0]),
                "model.context_encoder.weight": torch.tensor([3.0]),
                "unrelated": torch.tensor([99.0]),
            },
            "epoch": 39,
            "global_step": 1234,
        },
        checkpoint_path,
    )
    model = DPRBiEncoder(question_encoder=TinyEncoder(0.0), context_encoder=TinyEncoder(0.0))

    epoch, step = load_nq_tower_weights(model, str(checkpoint_path))

    assert (epoch, step) == (39, 1234)
    assert model.question_encoder.weight.item() == 2.0
    assert model.context_encoder.weight.item() == 3.0


def test_max_document_scores_uses_best_chunk_per_document():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    chunks = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])

    scores = max_document_scores(queries, chunks, torch.tensor([0, 0, 1]), doc_count=2)

    assert scores.tolist() == [[2.0, 0.0], [0.0, 3.0]]


def test_masked_positive_loss_excludes_other_known_positives_and_backpropagates():
    scores = torch.tensor([[3.0, 100.0, 1.0]], requires_grad=True)
    known_positives = torch.tensor([[True, True, False]])

    loss = masked_positive_loss(
        scores,
        doc_indices=torch.tensor([10, 11, 12]),
        targets=torch.tensor([10]),
        known_positive_mask=known_positives,
        weights=torch.tensor([2.0]),
    )
    expected = F.cross_entropy(torch.tensor([[3.0, 1.0]]), torch.tensor([0])) * 2
    loss.backward()

    assert loss.item() == pytest.approx(expected.item())
    assert scores.grad[0, 1].item() == 0.0
    assert torch.count_nonzero(scores.grad).item() == 2


def test_macro_retrieval_metrics_skips_unjudged_queries_and_uses_linear_gain():
    scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

    recall, ndcg, skipped = macro_retrieval_metrics(
        scores,
        query_ids=["judged", "unjudged"],
        doc_ids=["d0", "d1", "d2"],
        relevance={"judged": {"d0": 2, "d2": 1}},
    )

    expected_ndcg = (2.0 + 1.0 / 2.0) / (2.0 + 1.0 / torch.log2(torch.tensor(3.0)).item())
    assert recall == 1.0
    assert ndcg == pytest.approx(expected_ndcg)
    assert skipped == ["unjudged"]


def test_rank_zero_metrics_are_broadcast_to_other_ranks(monkeypatch):
    broadcast_values = torch.tensor([0.25, 0.5], dtype=torch.float64)

    def fake_broadcast(metrics, src):
        assert src == 0
        metrics.copy_(broadcast_values)

    module = "smile_lightning.apps.retrieval.nfcorpus.module.dist"
    monkeypatch.setattr(f"{module}.is_initialized", lambda: True)
    monkeypatch.setattr(f"{module}.get_world_size", lambda: 8)
    monkeypatch.setattr(f"{module}.get_rank", lambda: 3)
    monkeypatch.setattr(f"{module}.broadcast", fake_broadcast)

    recall, ndcg = broadcast_rank_zero_metrics(None, None, torch.device("cpu"))

    assert recall.item() == 0.25
    assert ndcg.item() == 0.5


def test_rank_zero_metrics_require_values_and_work_without_distributed():
    recall, ndcg = broadcast_rank_zero_metrics(0.2, 0.4, torch.device("cpu"))

    assert recall.item() == pytest.approx(0.2)
    assert ndcg.item() == pytest.approx(0.4)
    with pytest.raises(ValueError, match="Rank zero must provide"):
        broadcast_rank_zero_metrics(None, None, torch.device("cpu"))


def test_similarity_vectors_preserves_dot_and_normalizes_cosine():
    vectors = torch.tensor([[3.0, 4.0], [0.0, 2.0]], requires_grad=True)

    assert similarity_vectors(vectors, "dot") is vectors
    cosine = similarity_vectors(vectors, "cosine")
    assert torch.allclose(cosine, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))
    assert torch.allclose(torch.linalg.vector_norm(cosine, dim=1), torch.ones(2))
    cosine[:, 0].sum().backward()
    assert vectors.grad is not None
    with pytest.raises(ValueError, match="Unsupported"):
        similarity_vectors(vectors, "euclidean")


def test_cosine_mode_is_applied_in_training_and_full_corpus_dev(monkeypatch):
    monkeypatch.setattr(
        "smile_lightning.apps.retrieval.nfcorpus.module.load_nq_tower_weights",
        lambda model, checkpoint_path: (39, 1234),
    )
    model = StaticBiEncoder()
    module = NFCorpusLightningModule(model, "unused.ckpt", similarity_mode="cosine", score_scale=20.0)
    data = SimpleNamespace(
        train_doc_ids=["d0", "d1"],
        train_query_ids=["q"],
        train_positive_sets={"q": {"d1"}},
        train_positive_counts={"q": 1},
        train_dataset=[0],
        dev_document_batches=lambda batch_size: iter([({}, [0, 1])]),
        val_batch_size=2,
        dev_query_ids=["q"],
        dev_query_id_to_index={"q": 0},
        dev_doc_ids=["d0", "d1"],
        dev_relevance={"q": {"d1": 1}},
    )
    module._trainer = SimpleNamespace(datamodule=data, is_global_zero=True)
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.__setitem__(name, value))
    batch = {
        "question": {},
        "context": {},
        "chunk_global_doc_indices": torch.tensor([0, 1]),
        "query_indices": torch.tensor([0]),
        "target_doc_indices": torch.tensor([1]),
    }

    loss = module.training_step(batch, 0)
    expected = F.cross_entropy(torch.tensor([[0.8, 1.0]]) * 20.0, torch.tensor([1]))
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert model.contexts.grad is not None
    module.on_validation_epoch_start()
    assert torch.allclose(torch.linalg.vector_norm(module._dev_context_vectors, dim=1), torch.ones(2))
    module.validation_step({"question": {}, "query_ids": ["q"]}, 0)
    assert torch.allclose(torch.linalg.vector_norm(module._dev_queries["q"]), torch.tensor(1.0))
    module.on_validation_epoch_end()
    assert logged["val/recall_at_10"].item() == 1.0
    assert logged["val/ndcg_at_10"].item() == 1.0

    with pytest.raises(ValueError, match="similarity_mode"):
        NFCorpusLightningModule(model, "unused.ckpt", similarity_mode="euclidean")
