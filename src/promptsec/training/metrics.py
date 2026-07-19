"""Fixed-vocabulary single-label, multi-label, and verdict diagnostics."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

CORE_HEADS = (
    "prompt_injection_verdict",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "instruction_presentation",
)


def single_label_metrics(
    truth: Sequence[int], predictions: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=ids, zero_division=0
    )
    macro = precision_recall_fscore_support(
        truth, predictions, labels=ids, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth, predictions, labels=ids, average="weighted", zero_division=0
    )
    present_recall = [recall[index] for index, value in enumerate(support) if value]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mcc = float(matthews_corrcoef(truth, predictions)) if truth else None
    return {
        "records": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(np.mean(present_recall)) if present_recall else None,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "matthews_correlation_coefficient": mcc if mcc is None or np.isfinite(mcc) else None,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=ids).tolist(),
        "labels": list(labels),
    }


def multilabel_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: Sequence[str],
    threshold: float | Sequence[float] = 0.5,
) -> dict[str, Any]:
    thresholds = np.asarray(threshold, dtype=float)
    predictions = (probabilities >= thresholds).astype(int)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        truth, predictions, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        average_precision = average_precision_score(truth, probabilities, average=None)
    return {
        "records": int(truth.shape[0]),
        "micro_precision": float(precision_micro),
        "micro_recall": float(recall_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "samples_f1": float(f1_score(truth, predictions, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(truth, predictions)),
        "per_label_average_precision": {
            label: (
                float(average_precision[index]) if np.isfinite(average_precision[index]) else None
            )
            for index, label in enumerate(labels)
        },
        "mean_average_precision": float(np.nanmean(average_precision)),
        "per_label_support": {
            label: int(truth[:, index].sum()) for index, label in enumerate(labels)
        },
        "thresholds": thresholds.tolist() if thresholds.ndim else float(thresholds),
    }


def select_multilabel_thresholds(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    per_label: bool = False,
) -> float | list[float]:
    candidates = np.arange(0.1, 0.91, 0.05)
    if not per_label:
        return float(
            max(
                candidates,
                key=lambda value: f1_score(
                    truth, probabilities >= value, average="macro", zero_division=0
                ),
            )
        )
    return [
        float(
            max(
                candidates,
                key=lambda value: f1_score(
                    truth[:, index], probabilities[:, index] >= value, zero_division=0
                ),
            )
        )
        for index in range(truth.shape[1])
    ]


def core_macro_f1(head_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    return float(np.mean([head_metrics[head]["macro_f1"] for head in CORE_HEADS]))


def verdict_diagnostics(
    truth: Sequence[int], probabilities: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    detected = labels.index("DETECTED")
    binary_truth = np.asarray(truth) == detected
    detected_scores = probabilities[:, detected]
    predictions = probabilities.argmax(axis=1)
    false_positive = np.logical_and(~binary_truth, predictions == detected)
    false_negative = np.logical_and(binary_truth, predictions != detected)
    precision, recall, thresholds = precision_recall_curve(binary_truth, detected_scores)
    try:
        auc = float(roc_auc_score(binary_truth, detected_scores))
    except ValueError:
        auc = None
    hist, edges = np.histogram(detected_scores, bins=np.linspace(0, 1, 11))
    return {
        "false_positive_rate": float(false_positive.sum() / max(1, (~binary_truth).sum())),
        "false_negative_rate": float(false_negative.sum() / max(1, binary_truth.sum())),
        "roc_auc": auc,
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "confidence_histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
    }
