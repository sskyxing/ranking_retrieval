"""The independent question and context BERT towers used by DPR."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


class DPRBiEncoder(nn.Module):
    """Two independent Transformer encoders with raw CLS pooling.

    No projection, normalization, or score temperature is applied. Supplying
    encoder objects is useful for tests; production configs normally supply a
    Hugging Face model name or local model directory.
    """

    def __init__(
        self,
        question_model_name_or_path: str = "bert-base-uncased",
        context_model_name_or_path: Optional[str] = None,
        question_encoder: Optional[nn.Module] = None,
        context_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        context_name = context_model_name_or_path or question_model_name_or_path

        if question_encoder is None:
            from transformers import AutoModel

            question_encoder = AutoModel.from_pretrained(question_model_name_or_path)
        if context_encoder is None:
            if context_model_name_or_path is None and question_encoder is not None:
                context_encoder = copy.deepcopy(question_encoder)
            else:
                from transformers import AutoModel

                context_encoder = AutoModel.from_pretrained(context_name)

        if question_encoder is context_encoder:
            raise ValueError("DPR question and context encoders must be independent module instances.")
        shared_parameters = {id(parameter) for parameter in question_encoder.parameters()} & {
            id(parameter) for parameter in context_encoder.parameters()
        }
        if shared_parameters:
            raise ValueError("DPR question and context encoders cannot share parameters.")
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

    @staticmethod
    def _raw_cls(output: Any) -> Tensor:
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, Mapping):
            hidden = output["last_hidden_state"]
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            raise TypeError("Transformer output does not expose last_hidden_state.")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("last_hidden_state must have shape [batch, sequence, hidden].")
        return hidden[:, 0, :]

    @classmethod
    def _encode(
        cls,
        encoder: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        output = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return cls._raw_cls(output)

    def encode_questions(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tensor:
        return self._encode(self.question_encoder, input_ids, attention_mask, token_type_ids)

    def encode_contexts(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tensor:
        return self._encode(self.context_encoder, input_ids, attention_mask, token_type_ids)

    def forward(
        self,
        question_input_ids: Tensor,
        question_attention_mask: Tensor,
        question_token_type_ids: Tensor,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        context_token_type_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return (
            self.encode_questions(question_input_ids, question_attention_mask, question_token_type_ids),
            self.encode_contexts(context_input_ids, context_attention_mask, context_token_type_ids),
        )
