"""Validation-only multilabel calibration and robust checkpoint scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def calibrate_per_label_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    candidates: Sequence[float] | None = None,
    minimum_positive_support: int = 2,
    minimum_negative_support: int = 2,
    fallback: float = 0.5,
) -> dict[str, Any]:
    """Choose deterministic per-label F1 thresholds on validation only."""

    grid = tuple(candidates or np.arange(0.10, 0.901, 0.05).round(2).tolist())
    selected: list[float] = []
    rows: dict[str, Any] = {}
    for index, label in enumerate(labels):
        target = truth[:, index].astype(int)
        positive = int(target.sum())
        negative = int(len(target) - positive)
        baseline_prediction = probabilities[:, index] >= fallback
        baseline_f1 = float(f1_score(target, baseline_prediction, zero_division=0))
        supported = positive >= minimum_positive_support and negative >= minimum_negative_support
        choice = fallback
        reason = "insufficient_validation_support"
        if supported:
            viable = [value for value in grid if not bool(np.all(probabilities[:, index] >= value))]
            if viable:
                choice = max(
                    viable,
                    key=lambda value: (
                        f1_score(target, probabilities[:, index] >= value, zero_division=0),
                        -abs(value - fallback),
                        value,
                    ),
                )
                reason = "validation_per_label_f1"
        prediction = probabilities[:, index] >= choice
        selected.append(float(choice))
        rows[label] = {
            "threshold": float(choice),
            "positive_support": positive,
            "negative_support": negative,
            "f1_before": baseline_f1,
            "f1_after": float(f1_score(target, prediction, zero_division=0)),
            "predicted_positive_rate": float(prediction.mean()),
            "provenance": reason,
        }
    return {
        "selection_split": "validation",
        "test_metrics_used": False,
        "grid": list(grid),
        "fallback": fallback,
        "thresholds": selected,
        "labels": rows,
    }


def robust_validation_score(
    *,
    original_core_macro_f1: float,
    verdict_macro_f1: float,
    hard_negative_false_positive_rate: float,
    counterfactual_sensitivity: float | None,
    multilabel_macro_f1: float | None,
) -> dict[str, Any]:
    """Fixed v0.2 validation formula; unavailable benefits are renormalized."""

    benefits: list[tuple[str, float, float | None]] = [
        ("original_core_macro_f1", 0.50, original_core_macro_f1),
        ("verdict_macro_f1", 0.20, verdict_macro_f1),
        ("validation_counterfactual_sensitivity", 0.15, counterfactual_sensitivity),
        ("multilabel_macro_f1", 0.15, multilabel_macro_f1),
    ]
    present = [(name, weight, value) for name, weight, value in benefits if value is not None]
    scale = sum(weight for _, weight, _ in present)
    positive = sum(weight * float(value) for _, weight, value in present) / max(scale, 1e-12)
    penalty = 0.25 * hard_negative_false_positive_rate
    return {
        "robust_validation_score": positive - penalty,
        "original_core_macro_f1": original_core_macro_f1,
        "formula": (
            "renormalized(0.50*core + 0.20*verdict + 0.15*validation_cf_sensitivity "
            "+ 0.15*multilabel_macro_f1) - 0.25*hard_negative_fpr"
        ),
        "components": {name: value for name, _, value in benefits},
        "benefit_weight_normalizer": scale,
        "overdefense_penalty": penalty,
        "test_metrics_used": False,
    }


def assert_selection_is_validation_only(selection: Mapping[str, Any]) -> None:
    if selection.get("test_metrics_used") is not False:
        raise ValueError("model selection must explicitly exclude test metrics")
    if selection.get("selection_split", "validation") != "validation":
        raise ValueError("model selection must use validation only")
