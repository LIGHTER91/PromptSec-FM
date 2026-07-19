"""Training-derived class weights and normalized nine-head loss."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from promptsec.training.labels import MULTILABEL_HEADS, LabelMapping


def compute_training_class_weights(
    records: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, LabelMapping],
    *,
    single_clip: tuple[float, float] = (0.25, 4.0),
    positive_clip: tuple[float, float] = (0.25, 10.0),
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    class_weights: dict[str, Any] = {}
    positive_weights: dict[str, Any] = {}
    for head, mapping in mappings.items():
        parent = "derived" if head == "prompt_injection_verdict" else "annotations"
        if mapping.multilabel:
            positives = Counter(label for record in records for label in record[parent][head])
            values = []
            for label in mapping.labels:
                positive = positives[label]
                negative = len(records) - positive
                raw = negative / positive if positive else positive_clip[1]
                values.append(min(positive_clip[1], max(positive_clip[0], raw)))
            positive_weights[head] = torch.tensor(values, dtype=torch.float32)
        else:
            counts = Counter(record[parent][head] for record in records)
            values = []
            classes = max(1, len(mapping.labels))
            for label in mapping.labels:
                count = counts[label]
                raw = len(records) / (classes * count) if count else 1.0
                values.append(min(single_clip[1], max(single_clip[0], raw)))
            class_weights[head] = torch.tensor(values, dtype=torch.float32)
    return class_weights, positive_weights


def compute_multitask_loss(
    logits: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    class_weights: Mapping[str, Any] | None = None,
    positive_weights: Mapping[str, Any] | None = None,
    head_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Use mean-reduced CE/BCE per head, then a documented weighted mean."""

    import torch
    import torch.nn.functional as functional

    class_weights = class_weights or {}
    positive_weights = positive_weights or {}
    head_weights = head_weights or {}
    losses: dict[str, Any] = {}
    weighted = []
    weights = []
    for head, values in logits.items():
        if head in MULTILABEL_HEADS:
            pos_weight = positive_weights.get(head)
            if pos_weight is not None:
                pos_weight = pos_weight.to(values.device)
            loss = functional.binary_cross_entropy_with_logits(
                values,
                labels[head].to(values.device, dtype=torch.float32),
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            weight = class_weights.get(head)
            if weight is not None:
                weight = weight.to(values.device)
            loss = functional.cross_entropy(
                values,
                labels[head].to(values.device, dtype=torch.long),
                weight=weight,
            )
        factor = float(head_weights.get(head, 1.0))
        losses[head] = loss
        weighted.append(loss * factor)
        weights.append(factor)
    if not weighted or sum(weights) <= 0:
        raise ValueError("at least one positive-weight head is required")
    total = torch.stack(weighted).sum() / sum(weights)
    return total, losses
