"""Training-derived class weights and normalized nine-head loss."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from promptsec.training.labels import MULTILABEL_HEADS, LabelMapping

WEIGHTED_CROSS_ENTROPY = "WEIGHTED_CROSS_ENTROPY"
FOCAL_CROSS_ENTROPY = "FOCAL_CROSS_ENTROPY"
WEIGHTED_BCE = "WEIGHTED_BCE"
ASYMMETRIC_FOCAL_BCE = "ASYMMETRIC_FOCAL_BCE"


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
    single_label_loss_mode: str = WEIGHTED_CROSS_ENTROPY,
    multilabel_loss_mode: str = WEIGHTED_BCE,
    focal_gamma: float = 2.0,
    gamma_positive: float = 1.0,
    gamma_negative: float = 4.0,
    probability_clip: float = 0.05,
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
            target = labels[head].to(values.device, dtype=torch.float32)
            if multilabel_loss_mode == WEIGHTED_BCE:
                loss = (
                    functional.binary_cross_entropy_with_logits(
                        values, target, pos_weight=pos_weight, reduction="none"
                    )
                    .mean(dim=-1)
                    .mean()
                )
            elif multilabel_loss_mode == ASYMMETRIC_FOCAL_BCE:
                loss = asymmetric_focal_bce_with_logits(
                    values,
                    target,
                    positive_weights=pos_weight,
                    gamma_positive=gamma_positive,
                    gamma_negative=gamma_negative,
                    probability_clip=probability_clip,
                )
            else:
                raise ValueError(f"unknown multilabel loss mode: {multilabel_loss_mode}")
        else:
            weight = class_weights.get(head)
            if weight is not None:
                weight = weight.to(values.device)
            target = labels[head].to(values.device, dtype=torch.long)
            if single_label_loss_mode == WEIGHTED_CROSS_ENTROPY:
                loss = functional.cross_entropy(values, target, weight=weight)
            elif single_label_loss_mode == FOCAL_CROSS_ENTROPY:
                per_record = functional.cross_entropy(
                    values, target, weight=weight, reduction="none"
                )
                probability = torch.softmax(values, dim=-1).gather(1, target[:, None]).squeeze(1)
                loss = (((1 - probability).clamp_min(0) ** focal_gamma) * per_record).mean()
            else:
                raise ValueError(f"unknown single-label loss mode: {single_label_loss_mode}")
        factor = float(head_weights.get(head, 1.0))
        losses[head] = loss
        weighted.append(loss * factor)
        weights.append(factor)
    if not weighted or sum(weights) <= 0:
        raise ValueError("at least one positive-weight head is required")
    total = torch.stack(weighted).sum() / sum(weights)
    return total, losses


def asymmetric_focal_bce_with_logits(
    logits: Any,
    targets: Any,
    *,
    positive_weights: Any | None = None,
    gamma_positive: float = 1.0,
    gamma_negative: float = 4.0,
    probability_clip: float = 0.05,
) -> Any:
    """Stable normalized asymmetric focal BCE, including all-zero rows."""

    import torch
    import torch.nn.functional as functional

    if gamma_positive < 0 or gamma_negative < 0 or not 0 <= probability_clip < 1:
        raise ValueError("invalid asymmetric focal parameters")
    targets = targets.to(logits.device, dtype=logits.dtype)
    base = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    positive_probability = probabilities
    negative_probability = (1 - probabilities + probability_clip).clamp(max=1.0)
    modulator = targets * (1 - positive_probability) ** gamma_positive
    modulator += (1 - targets) * (1 - negative_probability) ** gamma_negative
    if positive_weights is not None:
        weights = 1 + targets * (positive_weights.to(logits.device) - 1)
        base = base * weights
    # Normalize each record by vocabulary size, not by number of positives.
    return (base * modulator).mean(dim=-1).mean()
