import json
from pathlib import Path
from typing import List, Optional

import pytest
import torch

from smile_lightning.apps.retrieval.dpr.data import (
    DPRCollator,
    DPRDataset,
    DPRPassage,
    DPRTrainingItem,
    DPRValidationItem,
    normalize_question,
    parse_nq_examples,
)


class FakeTokenizer:
    """Tiny tokenizer that makes pair order and token types observable."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        first: List[str],
        second: Optional[List[str]] = None,
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ):
        self.calls.append((list(first), None if second is None else list(second)))
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        token_types = []
        for index, _ in enumerate(first):
            if second is None:
                row = [101, 10 + index, 102]
                types = [0, 0, 0]
            else:
                row = [101, 20 + index, 102, 30 + index, 102]
                # This is the modern tokenizer default that DPRCollator must override.
                types = [0, 0, 0, 1, 1]
            rows.append((row + [0] * max_length)[:max_length])
            token_types.append((types + [0] * max_length)[:max_length])
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0).long(),
            "token_type_ids": torch.tensor(token_types),
        }


def _nq_data():
    return [
        {
            "question": "Who wrote Hamlet?",
            "positive_ctxs": [
                {"title": "William Shakespeare", "text": "Shakespeare wrote Hamlet."},
                {"title": "Unused positive", "text": "This is not selected."},
            ],
            "hard_negative_ctxs": [
                {"title": "Hard zero", "text": "Zero"},
                {"title": "Hard one", "text": "One"},
            ],
            "negative_ctxs": [
                {"title": "Ordinary zero", "text": "Ordinary"},
            ],
        }
    ]


def test_schema_requires_top_level_array_and_positive_passages():
    with pytest.raises(ValueError, match="top-level JSON array"):
        parse_nq_examples({"data": _nq_data()})
    with pytest.raises(ValueError, match="positive_ctxs.*at least one"):
        parse_nq_examples([{**_nq_data()[0], "positive_ctxs": []}])
    with pytest.raises(ValueError, match="hard_negative_ctxs.*JSON array"):
        parse_nq_examples([{**_nq_data()[0], "hard_negative_ctxs": None}])
    with pytest.raises(ValueError, match=r"positive_ctxs\[0\].title"):
        parse_nq_examples([{**_nq_data()[0], "positive_ctxs": [{"text": "missing title"}]}])


def test_dataset_uses_first_positive_and_deterministic_validation(tmp_path: Path):
    path = tmp_path / "nq.json"
    path.write_text(json.dumps(_nq_data()), encoding="utf-8")

    validation_item = DPRDataset(path, training=False)[0]

    assert validation_item.question == "Who wrote Hamlet"
    assert validation_item.positive.title == "William Shakespeare"
    assert validation_item.hard_negative.title == "Hard zero"
    assert [context.title for context in validation_item.contexts] == [
        "William Shakespeare",
        "Hard zero",
        "Hard one",
        "Ordinary zero",
    ]


def test_training_randomly_selects_one_hard_negative(tmp_path: Path, monkeypatch):
    path = tmp_path / "nq.json"
    path.write_text(json.dumps(_nq_data()), encoding="utf-8")
    selected_lengths = []

    def choose_last(passages):
        selected_lengths.append(len(passages))
        return passages[-1]

    monkeypatch.setattr("smile_lightning.apps.retrieval.dpr.data.random.choice", choose_last)
    training_item = DPRDataset(path, training=True)[0]

    assert selected_lengths == [2]
    assert training_item.hard_negative.title == "Hard one"


def test_empty_hard_negative_keeps_positive_for_train_and_val(tmp_path: Path):
    path = tmp_path / "nq.json"
    path.write_text(json.dumps([{**_nq_data()[0], "hard_negative_ctxs": []}]), encoding="utf-8")

    training_item = DPRDataset(path, training=True)[0]
    validation_item = DPRDataset(path, training=False)[0]

    assert training_item.hard_negative is None
    assert [context.title for context in training_item.contexts] == ["William Shakespeare"]
    assert validation_item.hard_negative is None
    assert [context.title for context in validation_item.contexts] == ["William Shakespeare", "Ordinary zero"]


@pytest.mark.parametrize(
    "question,expected",
    [("Question?", "Question"), ("Question??", "Question?"), ("Question", "Question")],
)
def test_normalize_question(question, expected):
    assert normalize_question(question) == expected


def test_collator_interleaves_title_text_pairs_and_forces_all_zero_segments():
    tokenizer = FakeTokenizer()
    items = [
        DPRTrainingItem(
            question="q0",
            positive=DPRPassage("p0 title", "p0 text"),
            hard_negative=DPRPassage("n0 title", "n0 text"),
        ),
        DPRTrainingItem(
            question="q1",
            positive=DPRPassage("p1 title", "p1 text"),
            hard_negative=DPRPassage("n1 title", "n1 text"),
        ),
    ]

    batch = DPRCollator(tokenizer, max_question_length=4, max_context_length=6)(items)

    assert tokenizer.calls == [
        (["q0", "q1"], None),
        (
            ["p0 title", "n0 title", "p1 title", "n1 title"],
            ["p0 text", "n0 text", "p1 text", "n1 text"],
        ),
    ]
    assert batch["positive_indices"].tolist() == [0, 2]
    assert batch["nll_context_indices"].tolist() == [0, 1, 2, 3]
    assert batch["nll_positive_indices"].tolist() == [0, 2]
    assert batch["question_input_ids"].shape == (2, 4)
    assert batch["context_input_ids"].shape == (4, 6)
    assert torch.count_nonzero(batch["question_token_type_ids"]) == 0
    assert torch.count_nonzero(batch["context_token_type_ids"]) == 0


def test_validation_collator_tracks_positive_offsets_for_variable_pools():
    tokenizer = FakeTokenizer()
    validation_items = [
        DPRValidationItem(
            question="q0",
            contexts=[
                DPRPassage("p0", "p0"),
                DPRPassage("h0", "h0"),
                DPRPassage("o0", "o0"),
            ],
        ),
        DPRValidationItem(
            question="q1",
            contexts=[DPRPassage("p1", "p1"), DPRPassage("h1", "h1")],
        ),
    ]

    batch = DPRCollator(tokenizer, max_question_length=4, max_context_length=6)(validation_items)

    assert batch["positive_indices"].tolist() == [0, 3]
    assert batch["nll_context_indices"].tolist() == [0, 1, 3, 4]
    assert batch["nll_positive_indices"].tolist() == [0, 2]
    assert batch["context_input_ids"].shape[0] == 5


def test_collator_tracks_positive_offsets_when_hard_negative_is_missing():
    tokenizer = FakeTokenizer()
    items = [
        DPRTrainingItem("q0", DPRPassage("p0", "p0"), None),
        DPRTrainingItem("q1", DPRPassage("p1", "p1"), DPRPassage("h1", "h1")),
    ]

    batch = DPRCollator(tokenizer)(items)

    assert tokenizer.calls[1][0] == ["p0", "p1", "h1"]
    assert batch["positive_indices"].tolist() == [0, 1]
    assert batch["nll_context_indices"].tolist() == [0, 1, 2]
    assert batch["nll_positive_indices"].tolist() == [0, 1]
    assert batch["context_input_ids"].shape[0] == 3


def test_validation_nll_uses_only_available_hard_negatives():
    items = [
        DPRValidationItem("q0", [DPRPassage("p0", "p0"), DPRPassage("o0", "o0")], hard_negative_index=None),
        DPRValidationItem("q1", [DPRPassage("p1", "p1"), DPRPassage("h1", "h1")]),
    ]

    batch = DPRCollator(FakeTokenizer())(items)

    assert batch["positive_indices"].tolist() == [0, 2]
    assert batch["nll_context_indices"].tolist() == [0, 2, 3]
    assert batch["nll_positive_indices"].tolist() == [0, 1]
