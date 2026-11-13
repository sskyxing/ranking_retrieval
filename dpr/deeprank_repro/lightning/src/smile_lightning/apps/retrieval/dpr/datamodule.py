"""PyTorch Lightning data module for DPR NQ training files."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .data import DPRCollator, DPRDataset


def _seed_worker(worker_id: int) -> None:
    """Give Python/NumPy random selection the seed assigned by DataLoader."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DPRDataModule(LightningDataModule):
    """Load train/validation NQ JSON arrays and tokenize them on collation."""

    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Optional[Union[str, Path]] = None,
        test_path: Optional[Union[str, Path]] = None,
        tokenizer_name_or_path: str = "bert-base-uncased",
        use_fast_tokenizer: bool = False,
        tokenizer: Optional[Any] = None,
        batch_size: int = 16,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        max_question_length: int = 256,
        max_context_length: int = 256,
        validation_num_hard_negatives: int = 30,
        validation_num_other_negatives: int = 30,
        validation_max_questions: int = 10000,
        seed: int = 0,
        drop_last: bool = True,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or (val_batch_size is not None and val_batch_size <= 0):
            raise ValueError("DPR batch sizes must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        self.save_hyperparameters(ignore=["tokenizer"])
        self.train_path = Path(train_path)
        self.val_path = Path(val_path) if val_path is not None else None
        self.test_path = Path(test_path) if test_path is not None else None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
        if self.persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0.")

        self.collator: Optional[DPRCollator] = None
        self.train_dataset: Optional[DPRDataset] = None
        self.val_dataset: Optional[DPRDataset] = None
        self.test_dataset: Optional[DPRDataset] = None
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.validation_num_hard_negatives = validation_num_hard_negatives
        self.validation_num_other_negatives = validation_num_other_negatives
        self.validation_max_questions = validation_max_questions

    def prepare_data(self) -> None:
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                use_fast=self.use_fast_tokenizer,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                use_fast=self.use_fast_tokenizer,
            )
        self.collator = DPRCollator(
            self.tokenizer,
            max_question_length=self.max_question_length,
            max_context_length=self.max_context_length,
        )

        if stage in (None, "fit"):
            self.train_dataset = DPRDataset(self.train_path, training=True)
            self.val_dataset = self._validation_dataset() if self.val_path is not None else None
        if stage in (None, "validate") and self.val_dataset is None and self.val_path is not None:
            self.val_dataset = self._validation_dataset()
        if stage in (None, "test") and self.test_path is not None:
            self.test_dataset = DPRDataset(
                self.test_path,
                training=False,
                validation_num_hard_negatives=self.validation_num_hard_negatives,
                validation_num_other_negatives=self.validation_num_other_negatives,
                max_examples=self.validation_max_questions,
            )

    def _validation_dataset(self) -> DPRDataset:
        if self.val_path is None:
            raise RuntimeError("Validation path is not configured.")
        return DPRDataset(
            self.val_path,
            training=False,
            validation_num_hard_negatives=self.validation_num_hard_negatives,
            validation_num_other_negatives=self.validation_num_other_negatives,
            max_examples=self.validation_max_questions,
        )

    def _loader(self, dataset: DPRDataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.collator is None:
            raise RuntimeError("DPRDataModule.setup() must run before requesting a dataloader.")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=_seed_worker,
            generator=generator,
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training data is not set up.")
        return self._loader(self.train_dataset, self.batch_size, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return self._loader(self.val_dataset, self.val_batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return self._loader(self.test_dataset, self.val_batch_size, shuffle=False, drop_last=False)
