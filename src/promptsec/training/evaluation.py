"""One-pass model inference and all-head metrics without payload logging."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from promptsec.training.labels import HEADS, MULTILABEL_HEADS, LabelMapping
from promptsec.training.metrics import (
    core_macro_f1,
    multilabel_metrics,
    single_label_metrics,
    verdict_diagnostics,
)


def predict_dataloader(model: Any, dataloader: Any, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    truth: dict[str, list[Any]] = defaultdict(list)
    logits: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    losses: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch["labels"]
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels={head: value.to(device) for head, value in labels.items()},
            )
            for head in HEADS:
                truth[head].extend(labels[head].cpu().tolist())
                logits[head].append(output.logits[head].detach().float().cpu().numpy())
                losses[head].append(float(output.head_losses[head].detach().cpu()))
            metadata.extend(batch["metadata"])
    return {
        "truth": dict(truth),
        "logits": {head: np.concatenate(values, axis=0) for head, values in logits.items()},
        "metadata": metadata,
        "per_head_loss": {head: float(np.mean(values)) for head, values in losses.items()},
    }


def metrics_from_predictions(
    predictions: Mapping[str, Any],
    mappings: Mapping[str, LabelMapping],
    thresholds: Mapping[str, float | Sequence[float]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for head, mapping in mappings.items():
        values = predictions["logits"][head]
        if head in MULTILABEL_HEADS:
            probabilities = 1 / (1 + np.exp(-values))
            metrics[head] = multilabel_metrics(
                np.asarray(predictions["truth"][head], dtype=int),
                probabilities,
                mapping.labels,
                thresholds.get(head, 0.5),
            )
        else:
            selected = values.argmax(axis=1).tolist()
            metrics[head] = single_label_metrics(
                predictions["truth"][head], selected, mapping.labels
            )
            if head == "prompt_injection_verdict":
                shifted = values - values.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                metrics[head]["verdict_diagnostics"] = verdict_diagnostics(
                    predictions["truth"][head], probabilities, mapping.labels
                )
    metrics["core_macro_f1"] = core_macro_f1(metrics)
    metrics["per_head_loss"] = predictions.get("per_head_loss", {})
    return metrics


def subset_predictions(predictions: Mapping[str, Any], indexes: Sequence[int]) -> dict[str, Any]:
    return {
        "truth": {
            head: [values[index] for index in indexes]
            for head, values in predictions["truth"].items()
        },
        "logits": {
            head: values[np.asarray(indexes)] for head, values in predictions["logits"].items()
        },
        "metadata": [predictions["metadata"][index] for index in indexes],
        "per_head_loss": {},
    }


def stratified_metrics(
    predictions: Mapping[str, Any],
    mappings: Mapping[str, LabelMapping],
    thresholds: Mapping[str, float | Sequence[float]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(predictions["metadata"]):
        groups[str(metadata[field])].append(index)
    return {
        value: metrics_from_predictions(
            subset_predictions(predictions, indexes), mappings, thresholds
        )
        for value, indexes in groups.items()
    }
