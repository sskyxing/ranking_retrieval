"""Natural Questions training data support for Dense Passage Retrieval."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DPRPassage:
    """A passage encoded as a BERT ``title, text`` pair."""

    title: str
    text: str


@dataclass(frozen=True)
class DPRExample:
    """The fields from an NQ DPR example that are needed for training."""

    question: str
    positive_ctxs: Sequence[DPRPassage]
    hard_negative_ctxs: Sequence[DPRPassage]
    negative_ctxs: Sequence[DPRPassage]


@dataclass(frozen=True)
class DPRTrainingItem:
    """One question with the passages selected for a single training step."""

    question: str
    positive: DPRPassage
    hard_negative: Optional[DPRPassage]

    @property
    def contexts(self) -> Sequence[DPRPassage]:
        return (self.positive, self.hard_negative) if self.hard_negative is not None else (self.positive,)

    @property
    def positive_index(self) -> int:
        return 0

    @property
    def hard_negative_index(self) -> Optional[int]:
        return 1 if self.hard_negative is not None else None


@dataclass(frozen=True)
class DPRValidationItem:
    """One question with its deterministic average-rank validation pool."""

    question: str
    contexts: Sequence[DPRPassage]
    positive_index: int = 0
    hard_negative_index: Optional[int] = 1

    @property
    def positive(self) -> DPRPassage:
        return self.contexts[self.positive_index]

    @property
    def hard_negative(self) -> Optional[DPRPassage]:
        return self.contexts[self.hard_negative_index] if self.hard_negative_index is not None else None


def normalize_question(question: str) -> str:
    """Match DPR preprocessing by removing one trailing question mark."""

    return question[:-1] if question.endswith("?") else question


def _require_list(value: Any, field: str, example_index: int, allow_empty: bool = False) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Example {example_index}: '{field}' must be a JSON array.")
    if not value and not allow_empty:
        raise ValueError(f"Example {example_index}: '{field}' must contain at least one passage.")
    return value


def _optional_list(value: Any, field: str, example_index: int) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Example {example_index}: '{field}' must be a JSON array.")
    return value


def _parse_passage(value: Any, field: str, example_index: int, passage_index: int) -> DPRPassage:
    if not isinstance(value, Mapping):
        raise ValueError(f"Example {example_index}: {field}[{passage_index}] must be a JSON object.")

    title = value.get("title")
    text = value.get("text")
    if not isinstance(title, str):
        raise ValueError(f"Example {example_index}: {field}[{passage_index}].title must be a string.")
    if not isinstance(text, str):
        raise ValueError(f"Example {example_index}: {field}[{passage_index}].text must be a string.")
    return DPRPassage(title=title, text=text)


def parse_nq_examples(data: Any) -> List[DPRExample]:
    """Validate and parse the DPR NQ top-level JSON-array schema."""

    if not isinstance(data, list):
        raise ValueError("DPR NQ data must be a top-level JSON array.")

    examples: List[DPRExample] = []
    for example_index, value in enumerate(data):
        if not isinstance(value, Mapping):
            raise ValueError(f"Example {example_index} must be a JSON object.")

        question = value.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError(f"Example {example_index}: 'question' must be a non-empty string.")

        positive_values = _require_list(value.get("positive_ctxs"), "positive_ctxs", example_index)
        hard_negative_values = _require_list(
            value.get("hard_negative_ctxs"), "hard_negative_ctxs", example_index, allow_empty=True
        )
        negative_values = _optional_list(value.get("negative_ctxs"), "negative_ctxs", example_index)
        positives = [
            _parse_passage(item, "positive_ctxs", example_index, passage_index)
            for passage_index, item in enumerate(positive_values)
        ]
        hard_negatives = [
            _parse_passage(item, "hard_negative_ctxs", example_index, passage_index)
            for passage_index, item in enumerate(hard_negative_values)
        ]
        negatives = [
            _parse_passage(item, "negative_ctxs", example_index, passage_index)
            for passage_index, item in enumerate(negative_values)
        ]
        examples.append(
            DPRExample(
                question=question,
                positive_ctxs=positives,
                hard_negative_ctxs=hard_negatives,
                negative_ctxs=negatives,
            )
        )
    return examples


def load_nq_examples(path: Union[str, Path]) -> List[DPRExample]:
    """Load a complete DPR NQ JSON array from disk."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return parse_nq_examples(json.load(stream))


class DPRDataset(Dataset):
    """Select DPR training pairs and deterministic validation pools.

    The first positive is always used. Training samples a hard negative using
    Python's worker-local RNG when one exists. Validation/test use the first
    available hard negative for NLL and retain configurable hard/ordinary
    negatives for average rank. Questions without a hard negative still
    contribute their positive passage to in-batch negative training.
    """

    def __init__(
        self,
        path: Union[str, Path],
        training: bool,
        validation_num_hard_negatives: int = 30,
        validation_num_other_negatives: int = 30,
        max_examples: int = 0,
    ) -> None:
        if validation_num_hard_negatives < 0 or validation_num_other_negatives < 0:
            raise ValueError("Validation negative counts cannot be negative.")
        if not training and validation_num_hard_negatives == 0:
            raise ValueError("Validation requires a positive hard-negative limit for NLL.")
        if max_examples < 0:
            raise ValueError("max_examples cannot be negative.")
        self.path = Path(path)
        self.training = training
        self.validation_num_hard_negatives = validation_num_hard_negatives
        self.validation_num_other_negatives = validation_num_other_negatives
        self.examples = load_nq_examples(self.path)
        if max_examples:
            self.examples = self.examples[:max_examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Union[DPRTrainingItem, DPRValidationItem]:
        example = self.examples[index]
        question = normalize_question(example.question)
        if self.training:
            return DPRTrainingItem(
                question=question,
                positive=example.positive_ctxs[0],
                hard_negative=random.choice(example.hard_negative_ctxs) if example.hard_negative_ctxs else None,
            )

        contexts = [example.positive_ctxs[0]]
        hard_negatives = example.hard_negative_ctxs[: self.validation_num_hard_negatives]
        contexts.extend(hard_negatives)
        contexts.extend(example.negative_ctxs[: self.validation_num_other_negatives])
        return DPRValidationItem(
            question=question, contexts=contexts, hard_negative_index=1 if hard_negatives else None
        )


class DPRCollator:
    """Tokenize questions plus training pairs or variable validation pools."""

    def __init__(self, tokenizer: Any, max_question_length: int = 256, max_context_length: int = 256) -> None:
        if max_question_length <= 0 or max_context_length <= 0:
            raise ValueError("Maximum sequence lengths must be positive.")
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length

    def _as_tensor_dict(self, encoded: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer output must contain input_ids.")
        result = {
            key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.long)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        input_ids = result["input_ids"]
        if "attention_mask" not in result:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = 0
            result["attention_mask"] = input_ids.ne(pad_token_id).long()
        # DPR's BERT tensorizer deliberately uses segment 0 for both sides of
        # a title/text pair. Modern tokenizers otherwise assign segment 1 to text.
        result["token_type_ids"] = torch.zeros_like(input_ids)
        return result

    def __call__(self, items: Sequence[Union[DPRTrainingItem, DPRValidationItem]]) -> Dict[str, torch.Tensor]:
        if not items:
            raise ValueError("Cannot collate an empty DPR batch.")

        questions = [item.question for item in items]
        passages: List[DPRPassage] = []
        positive_indices: List[int] = []
        nll_context_indices: List[int] = []
        nll_positive_indices: List[int] = []
        for item in items:
            context_offset = len(passages)
            if not item.contexts or not 0 <= item.positive_index < len(item.contexts):
                raise ValueError("Each DPR item must have a positive passage in its context pool.")
            hard_negative_index = item.hard_negative_index
            if hard_negative_index is not None and (
                not 0 <= hard_negative_index < len(item.contexts) or hard_negative_index == item.positive_index
            ):
                raise ValueError("A hard-negative index must identify another passage in the context pool.")
            positive_indices.append(context_offset + item.positive_index)
            nll_positive_indices.append(len(nll_context_indices))
            nll_context_indices.append(context_offset + item.positive_index)
            if hard_negative_index is not None:
                nll_context_indices.append(context_offset + hard_negative_index)
            passages.extend(item.contexts)
        titles = [passage.title for passage in passages]
        texts = [passage.text for passage in passages]

        question_tokens = self._as_tensor_dict(
            self.tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_question_length,
                return_tensors="pt",
            )
        )
        context_tokens = self._as_tensor_dict(
            self.tokenizer(
                titles,
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
            )
        )

        batch: Dict[str, torch.Tensor] = {}
        for key, value in question_tokens.items():
            batch[f"question_{key}"] = value
        for key, value in context_tokens.items():
            batch[f"context_{key}"] = value
        batch["positive_indices"] = torch.tensor(positive_indices, dtype=torch.long)
        batch["nll_context_indices"] = torch.tensor(nll_context_indices, dtype=torch.long)
        batch["nll_positive_indices"] = torch.tensor(nll_positive_indices, dtype=torch.long)
        return batch
