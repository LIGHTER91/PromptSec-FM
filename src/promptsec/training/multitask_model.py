"""Hugging Face-compatible shared XLM-R encoder with nine classification heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from promptsec.training.labels import MULTILABEL_HEADS
from promptsec.training.losses import compute_multitask_loss


class PromptSecMultitaskConfig(PretrainedConfig):
    model_type = "promptsec_xlmr_multitask"

    def __init__(
        self,
        *,
        encoder_config: dict[str, Any] | None = None,
        head_dimensions: dict[str, int] | None = None,
        label_mappings: dict[str, Any] | None = None,
        special_tokens: list[str] | None = None,
        dropout: float = 0.1,
        pooling: str = "first_contextual_token",
        preprocessing_version: str = "full-context-section-budget-v0.1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.head_dimensions = head_dimensions or {}
        self.label_mappings = label_mappings or {}
        self.special_tokens = special_tokens or []
        self.dropout = dropout
        self.pooling = pooling
        self.preprocessing_version = preprocessing_version


@dataclass
class PromptSecMultitaskOutput(ModelOutput):
    loss: torch.Tensor | None = None
    head_losses: dict[str, torch.Tensor] | None = None
    logits: dict[str, torch.Tensor] | None = None


class PromptSecMultitaskModel(PreTrainedModel):
    config_class = PromptSecMultitaskConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: PromptSecMultitaskConfig,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__(config)
        if encoder is None:
            values = dict(config.encoder_config)
            model_type = values.pop("model_type", "xlm-roberta")
            encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
        self.encoder = encoder
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleDict(
            {
                head: nn.Linear(hidden_size, dimensions)
                for head, dimensions in config.head_dimensions.items()
            }
        )
        self.class_weights: dict[str, torch.Tensor] = {}
        self.positive_weights: dict[str, torch.Tensor] = {}
        self.head_weights: dict[str, float] = {}
        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        model_name: str,
        *,
        head_dimensions: dict[str, int],
        label_mappings: dict[str, Any],
        special_tokens: list[str],
        dropout: float = 0.1,
    ) -> PromptSecMultitaskModel:
        encoder = AutoModel.from_pretrained(model_name)
        config = PromptSecMultitaskConfig(
            encoder_config=encoder.config.to_dict(),
            head_dimensions=head_dimensions,
            label_mappings=label_mappings,
            special_tokens=special_tokens,
            dropout=dropout,
        )
        return cls(config, encoder=encoder)

    def set_loss_configuration(
        self,
        *,
        class_weights: dict[str, torch.Tensor],
        positive_weights: dict[str, torch.Tensor],
        head_weights: dict[str, float],
    ) -> None:
        self.class_weights = class_weights
        self.positive_weights = positive_weights
        self.head_weights = head_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> PromptSecMultitaskOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled = self.dropout(encoded.last_hidden_state[:, 0, :])
        logits = {head: layer(pooled) for head, layer in self.heads.items()}
        if labels is None:
            return PromptSecMultitaskOutput(logits=logits)
        total, head_losses = compute_multitask_loss(
            logits,
            labels,
            class_weights=self.class_weights,
            positive_weights=self.positive_weights,
            head_weights=self.head_weights,
        )
        return PromptSecMultitaskOutput(loss=total, head_losses=head_losses, logits=logits)

    def resize_encoder_token_embeddings(self, size: int) -> Any:
        return self.encoder.resize_token_embeddings(size)

    def decode_predictions(
        self,
        logits: dict[str, torch.Tensor],
        *,
        threshold: float | Mapping[str, float | Sequence[float]] = 0.5,
    ) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        mappings = self.config.label_mappings
        for head, values in logits.items():
            labels = mappings[head]["labels"]
            if head in MULTILABEL_HEADS:
                probabilities = torch.sigmoid(values).detach()
                selected_threshold = (
                    threshold.get(head, 0.5) if isinstance(threshold, Mapping) else threshold
                )
                threshold_values = torch.as_tensor(selected_threshold, device=probabilities.device)
                selected = probabilities >= threshold_values
                decoded[head] = {
                    "labels": [
                        [label for label, keep in zip(labels, row, strict=True) if bool(keep)]
                        for row in selected
                    ],
                    "probabilities": probabilities.detach().cpu().tolist(),
                }
            else:
                probabilities = torch.softmax(values, dim=-1).detach()
                selected = probabilities.argmax(dim=-1).detach().cpu().tolist()
                decoded[head] = {
                    "labels": [labels[index] for index in selected],
                    "probabilities": probabilities.detach().cpu().tolist(),
                }
        return decoded
