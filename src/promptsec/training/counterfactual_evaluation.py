"""All-head counterfactual relation metrics and transition matrices."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from promptsec.training.labels import MULTILABEL_HEADS, LabelMapping


def _decoded_predictions(
    logits: np.ndarray,
    mapping: LabelMapping,
    threshold: float | Sequence[float],
) -> list[Any]:
    if mapping.multilabel:
        probabilities = 1 / (1 + np.exp(-logits))
        selected = probabilities >= np.asarray(threshold)
        return [
            tuple(label for label, keep in zip(mapping.labels, row, strict=True) if keep)
            for row in selected
        ]
    return [mapping.labels[index] for index in logits.argmax(axis=1)]


def evaluate_counterfactual_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    mappings: Mapping[str, LabelMapping],
    thresholds: Mapping[str, float | Sequence[float]],
) -> dict[str, Any]:
    by_id = {metadata["id"]: index for index, metadata in enumerate(predictions["metadata"])}
    results: dict[str, Any] = {}
    for head, mapping in mappings.items():
        decoded = _decoded_predictions(
            predictions["logits"][head], mapping, thresholds.get(head, 0.5)
        )
        parent = "derived" if head == "prompt_injection_verdict" else "annotations"
        groups: dict[str, list[tuple[Any, Any, Mapping[str, Any]]]] = defaultdict(list)
        for record in records:
            counterfactual = record["extensions"]["policybench_v0_1"]["counterfactual"]
            if not isinstance(counterfactual, Mapping):
                continue
            truth = record[parent][head]
            if head in MULTILABEL_HEADS:
                truth = tuple(truth)
            groups[counterfactual["counterfactual_group_id"]].append(
                (truth, decoded[by_id[record["id"]]], counterfactual)
            )
        rows = []
        transitions: Counter[str] = Counter()
        for group_id, members in groups.items():
            if len(members) != 2:
                continue
            left, right = members
            expected_change = any(
                item.get("field") == head
                for metadata in (left[2], right[2])
                for item in metadata.get("expected_label_changes", [])
            )
            true_change = left[0] != right[0]
            if expected_change != true_change:
                raise ValueError(f"counterfactual metadata mismatch: {group_id}/{head}")
            predicted_change = left[1] != right[1]
            transitions[f"{left[1]} -> {right[1]}"] += 1
            rows.append(
                {
                    "type": left[2]["counterfactual_type"],
                    "expected_change": expected_change,
                    "predicted_change": predicted_change,
                    "exact": left[0] == left[1] and right[0] == right[1],
                }
            )
        results[head] = _summarize(rows)
        results[head]["prediction_transition_matrix"] = dict(sorted(transitions.items()))
        results[head]["by_counterfactual_type"] = {
            kind: _summarize([row for row in rows if row["type"] == kind])
            for kind in (
                "POLICY_CHANGE",
                "USER_GOAL_CHANGE",
                "SOURCE_ROLE_CHANGE",
                "AUTHORITY_DELEGATION_CHANGE",
                "CAPABILITY_CHANGE",
                "PRESENTATION_CHANGE",
            )
        }
    return results


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["expected_change"]]
    invariant = [row for row in rows if not row["expected_change"]]
    return {
        "groups": len(rows),
        "pairwise_accuracy": _mean(
            [row["expected_change"] == row["predicted_change"] for row in rows]
        ),
        "expected_change_sensitivity": _mean([row["predicted_change"] for row in changed]),
        "invariant_prediction_consistency": _mean(
            [not row["predicted_change"] for row in invariant]
        ),
        "exact_group_accuracy": _mean([row["exact"] for row in rows]),
        "expected_change_groups": len(changed),
        "invariant_groups": len(invariant),
    }


def _mean(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None
