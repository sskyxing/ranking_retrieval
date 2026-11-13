from types import SimpleNamespace

import pytest
import torch
from torch import nn

from smile_lightning.apps.retrieval.dpr.model import DPRBiEncoder
from smile_lightning.apps.retrieval.dpr.module import (
    DPRLightningModule,
    _offset_rank_major_indices,
    concatenate_rank_shards_with_local_grad,
    dpr_nll,
    exact_average_rank,
    validate_equal_batch_sizes,
)


class FakeEncoder(nn.Module):
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = offset
        self.last_token_type_ids = None

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        assert return_dict is True
        self.last_token_type_ids = token_type_ids
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3) * self.scale + self.offset
        return SimpleNamespace(last_hidden_state=hidden)


def _tokens(rows: int):
    input_ids = torch.arange(1, rows * 2 + 1).view(rows, 2)
    return input_ids, torch.ones_like(input_ids), torch.zeros_like(input_ids)


def test_biencoder_has_independent_towers_and_returns_raw_cls():
    question_encoder = FakeEncoder(offset=0.0)
    model = DPRBiEncoder(question_encoder=question_encoder)
    question = _tokens(2)
    context = _tokens(4)

    q_vectors, context_vectors = model(*question, *context)

    assert model.question_encoder is not model.context_encoder
    assert q_vectors.shape == (2, 3)
    assert context_vectors.shape == (4, 3)
    assert torch.equal(q_vectors[0], torch.tensor([1.0, 1.0, 1.0]))
    # Raw CLS is deliberately neither a pooler output nor L2-normalized.
    assert q_vectors[0].norm().item() == pytest.approx(3 ** 0.5)


def test_biencoder_rejects_parameter_sharing_between_distinct_modules():
    shared = nn.Parameter(torch.tensor(1.0))
    question_encoder = FakeEncoder()
    context_encoder = FakeEncoder()
    question_encoder.scale = shared
    context_encoder.scale = shared

    with pytest.raises(ValueError, match="cannot share parameters"):
        DPRBiEncoder(question_encoder=question_encoder, context_encoder=context_encoder)


def test_dpr_nll_uses_even_positive_columns_and_backpropagates_to_both_towers():
    q_vectors = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    context_vectors = torch.tensor(
        [[2.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -1.0]],
        requires_grad=True,
    )

    loss, accuracy, logits = dpr_nll(q_vectors, context_vectors)
    loss.backward()

    assert logits.shape == (2, 4)
    assert accuracy.item() == 1.0
    assert q_vectors.grad is not None and torch.count_nonzero(q_vectors.grad) > 0
    assert context_vectors.grad is not None and torch.count_nonzero(context_vectors.grad) > 0


def test_local_rank_replacement_preserves_only_local_source_grad():
    local = torch.tensor([[10.0, 11.0]], requires_grad=True)
    gathered = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])

    global_tensor = concatenate_rank_shards_with_local_grad(gathered, local, rank=1)
    global_tensor.sum().backward()

    assert global_tensor.tolist() == [[1.0, 2.0], [10.0, 11.0], [5.0, 6.0]]
    assert local.grad.tolist() == [[1.0, 1.0]]


def test_equal_batch_size_validation_is_explicit():
    assert validate_equal_batch_sizes([8, 8, 8], "questions") == 8
    with pytest.raises(RuntimeError, match=r"equal local batch sizes.*\[8, 7\]"):
        validate_equal_batch_sizes([8, 7], "questions")


def test_positive_indices_offset_for_variable_rank_context_pools():
    local_indices = torch.tensor([0, 1, 0, 2])

    global_indices = _offset_rank_major_indices(local_indices, [2, 2], [3, 4])

    assert global_indices.tolist() == [0, 1, 3, 5]
    with pytest.raises(ValueError, match="question counts"):
        _offset_rank_major_indices(local_indices, [2, 1], [3, 4])
    with pytest.raises(ValueError, match="outside its context pool"):
        _offset_rank_major_indices(torch.tensor([0, 3, 0, 2]), [2, 2], [3, 4])


def test_uneven_rank_context_gather_drops_padding_and_keeps_local_grad(monkeypatch):
    local = torch.tensor([[10.0, 11.0], [12.0, 13.0]], requires_grad=True)
    remote = torch.tensor([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]])
    calls = []

    def fake_all_gather(outputs, value):
        calls.append(value.shape)
        if value.ndim == 1:
            outputs[0].fill_(2)
            outputs[1].fill_(3)
        else:
            outputs[0].copy_(value)
            outputs[1].copy_(remote)

    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.is_initialized", lambda: True)
    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.get_rank", lambda: 0)
    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.all_gather", fake_all_gather)

    gathered, counts = DPRLightningModule._gather_global(local, "context embeddings", allow_uneven=True)
    gathered.sum().backward()

    assert calls == [(1,), (3, 2)]
    assert counts == [2, 3]
    assert gathered.tolist() == [[10.0, 11.0], [12.0, 13.0], [20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]
    assert local.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_rank_major_global_order_keeps_even_positive_indices():
    local_rank_zero = torch.tensor([[10.0], [11.0]])
    rank_one = torch.tensor([[20.0], [21.0]])
    gathered = torch.stack((torch.tensor([[0.0], [1.0]]), rank_one), dim=0)

    global_contexts = concatenate_rank_shards_with_local_grad(gathered, local_rank_zero, rank=0)

    assert global_contexts.squeeze(1).tolist() == [10.0, 11.0, 20.0, 21.0]
    assert torch.arange(2).mul(2).tolist() == [0, 2]


def test_average_rank_is_exact_with_chunking():
    questions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # q0 positive has zero-based rank 1; q1 positive has rank 0. Mean = 0.5.
    contexts = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])

    average_rank = exact_average_rank(questions, contexts, context_chunk_size=1, query_chunk_size=1)

    assert average_rank.item() == 0.5


def test_average_rank_supports_variable_validation_context_pools():
    questions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    contexts = torch.tensor(
        [
            [0.5, 0.0],  # q0 positive (global context 0), rank 1
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],  # q1 positive (global context 4), rank 1
            [0.0, -1.0],
        ]
    )

    rank = exact_average_rank(
        questions, contexts, positive_indices=torch.tensor([0, 4]), context_chunk_size=2, query_chunk_size=1
    )

    assert rank.item() == 1.0


class OptimizerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.LayerNorm = nn.LayerNorm(2)

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        values = self.LayerNorm(self.linear(input_ids.float()))
        return SimpleNamespace(last_hidden_state=values.unsqueeze(1))


def test_adamw_no_decay_groups_and_linear_warmup():
    module = DPRLightningModule(
        DPRBiEncoder(question_encoder=OptimizerEncoder()),
        weight_decay=0.01,
        learning_rate=2e-5,
        warmup_steps=2,
        max_steps=10,
    )

    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.0]
    assert len(optimizer.param_groups[1]["params"]) > 0
    assert module._linear_warmup_decay(0, 2, 10) == 0.0
    assert module._linear_warmup_decay(2, 2, 10) == 1.0
    assert module._linear_warmup_decay(10, 2, 10) == 0.0
    assert scheduler is not None


def test_training_step_runs_with_fake_towers_and_backpropagates():
    model = DPRBiEncoder(question_encoder=FakeEncoder(), context_encoder=FakeEncoder(offset=0.25))
    module = DPRLightningModule(model, max_steps=2)
    module.log = lambda *args, **kwargs: None
    question = _tokens(2)
    context = _tokens(4)
    batch = {
        "question_input_ids": question[0],
        "question_attention_mask": question[1],
        "question_token_type_ids": question[2],
        "context_input_ids": context[0],
        "context_attention_mask": context[1],
        "context_token_type_ids": context[2],
        "positive_indices": torch.tensor([0, 2]),
        "nll_context_indices": torch.tensor([0, 1, 2, 3]),
        "nll_positive_indices": torch.tensor([0, 2]),
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.question_encoder.scale.grad is not None
    assert model.context_encoder.scale.grad is not None


def test_training_step_accepts_positive_only_question():
    model = DPRBiEncoder(question_encoder=FakeEncoder(), context_encoder=FakeEncoder(offset=0.25))
    module = DPRLightningModule(model, max_steps=2)
    module.log = lambda *args, **kwargs: None
    question = _tokens(2)
    context = _tokens(3)
    batch = {
        "question_input_ids": question[0],
        "question_attention_mask": question[1],
        "question_token_type_ids": question[2],
        "context_input_ids": context[0],
        "context_attention_mask": context[1],
        "context_token_type_ids": context[2],
        "positive_indices": torch.tensor([0, 1]),
        "nll_context_indices": torch.tensor([0, 1, 2]),
        "nll_positive_indices": torch.tensor([0, 1]),
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.question_encoder.scale.grad is not None
    assert model.context_encoder.scale.grad is not None


def test_training_step_compensates_for_ddp_gradient_average(monkeypatch):
    model = DPRBiEncoder(question_encoder=FakeEncoder(), context_encoder=FakeEncoder(offset=0.25))
    module = DPRLightningModule(model, max_steps=2)
    module._shared_step = lambda batch, stage: (torch.tensor(2.0), None, None, None)
    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.is_initialized", lambda: True)
    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.module.dist.get_world_size", lambda: 8)

    assert module.training_step({}, 0).item() == 16.0


def test_validation_nll_uses_only_positive_and_first_hard_negative_before_average_rank():
    model = DPRBiEncoder(question_encoder=FakeEncoder(), context_encoder=FakeEncoder(offset=0.25))
    module = DPRLightningModule(model, enable_average_rank=True, average_rank_start_epoch=30, max_steps=2)
    module.log = lambda *args, **kwargs: None
    module._trainer = SimpleNamespace(current_epoch=0)
    question = _tokens(2)
    context = _tokens(6)
    batch = {
        "question_input_ids": question[0],
        "question_attention_mask": question[1],
        "question_token_type_ids": question[2],
        "context_input_ids": context[0],
        "context_attention_mask": context[1],
        "context_token_type_ids": context[2],
        "positive_indices": torch.tensor([0, 3]),
        "nll_context_indices": torch.tensor([0, 1, 3, 4]),
        "nll_positive_indices": torch.tensor([0, 2]),
    }

    _, _, questions, encoded_contexts = module._shared_step(batch, "val")

    assert questions.shape[0] == 2
    assert encoded_contexts.shape[0] == 4


def test_validation_nll_accepts_positive_only_question():
    model = DPRBiEncoder(question_encoder=FakeEncoder(), context_encoder=FakeEncoder(offset=0.25))
    module = DPRLightningModule(model, max_steps=2)
    module.log = lambda *args, **kwargs: None
    question = _tokens(2)
    context = _tokens(4)
    batch = {
        "question_input_ids": question[0],
        "question_attention_mask": question[1],
        "question_token_type_ids": question[2],
        "context_input_ids": context[0],
        "context_attention_mask": context[1],
        "context_token_type_ids": context[2],
        "positive_indices": torch.tensor([0, 2]),
        "nll_context_indices": torch.tensor([0, 2, 3]),
        "nll_positive_indices": torch.tensor([0, 1]),
    }

    loss, _, questions, encoded_contexts = module._shared_step(batch, "val")

    assert torch.isfinite(loss)
    assert questions.shape[0] == 2
    assert encoded_contexts.shape[0] == 3
