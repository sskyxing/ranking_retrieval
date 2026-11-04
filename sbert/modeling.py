"""Shared BERT encoder and the NLI objective used by Sentence-BERT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Include non-padding tokens, including CLS/SEP, as in the original code."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class SBERTNLI(nn.Module):
    def __init__(self, model_name: str, max_seq_length: int = 130) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.classifier = nn.Linear(self.encoder.config.hidden_size * 3, 3)

    def tokenize(self, sentences: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device)
            for key, value in self.tokenizer(
                sentences, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            ).items()
        }

    def sentence_embeddings(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**features, return_dict=True)
        return mean_pool(output.last_hidden_state, features["attention_mask"])

    def forward(
        self, first: dict[str, torch.Tensor], second: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        u = self.sentence_embeddings(first)
        v = self.sentence_embeddings(second)
        return self.classifier(torch.cat((u, v, torch.abs(u - v)), dim=1))

    @torch.inference_mode()
    def encode(self, sentences: list[str], batch_size: int = 32) -> torch.Tensor:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        embeddings = []
        try:
            for offset in range(0, len(sentences), batch_size):
                features = self.tokenize(sentences[offset:offset + batch_size], device)
                embeddings.append(self.sentence_embeddings(features).cpu())
        finally:
            self.train(was_training)
        if not embeddings:
            return torch.empty((0, self.encoder.config.hidden_size))
        return torch.cat(embeddings)

    def export_sentence_transformer(
        self, directory: str | Path, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a standard Transformer + mean Pooling SentenceTransformer model."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)
        pooling = directory / "1_Pooling"
        pooling.mkdir(exist_ok=True)
        files = {
            "modules.json": [
                {"idx": 0, "name": "0", "path": "",
                 "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling",
                 "type": "sentence_transformers.models.Pooling"},
            ],
            "sentence_bert_config.json": {
                "max_seq_length": self.max_seq_length, "do_lower_case": True,
            },
            "1_Pooling/config.json": {
                "word_embedding_dimension": self.encoder.config.hidden_size,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            },
            "sbert_config.json": {
                "max_seq_length": self.max_seq_length, "pooling": "mean",
                "normalize_embeddings": False,
                "architecture": "shared-bert + mean-pooling",
                **(metadata or {}),
            },
        }
        for name, value in files.items():
            (directory / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
