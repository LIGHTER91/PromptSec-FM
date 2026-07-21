"""Descriptive English/French all-head comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compare_language_metrics(
    english: Mapping[str, Any], french: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "caveat": (
            "The official language-OOD split may differ from English evaluation data in more "
            "than language; differences are descriptive, not causal."
        ),
        "heads": {},
    }
    for head in english:
        if head in {"core_macro_f1", "per_head_loss"} or head not in french:
            continue
        en_value = english[head].get("macro_f1")
        fr_value = french[head].get("macro_f1")
        if en_value is None or fr_value is None:
            continue
        item = {
            "english_macro_f1": en_value,
            "french_macro_f1": fr_value,
            "difference_en_minus_fr": en_value - fr_value,
        }
        en_classes = english[head].get("per_class")
        fr_classes = french[head].get("per_class")
        if en_classes and fr_classes:
            item["class_f1_difference_en_minus_fr"] = {
                label: values["f1"] - fr_classes[label]["f1"]
                for label, values in en_classes.items()
            }
        if head == "prompt_injection_verdict":
            en_calibration = english[head].get("verdict_diagnostics", {})
            fr_calibration = french[head].get("verdict_diagnostics", {})
            item["calibration_en_vs_fr"] = {
                metric: {
                    "english": en_calibration.get(metric),
                    "french": fr_calibration.get(metric),
                }
                for metric in ("brier_score", "expected_calibration_error_10_bins")
            }
        if head in {"authority_status", "user_goal_alignment"}:
            item["terminology_or_relation_error_diagnostic"] = item.get(
                "class_f1_difference_en_minus_fr", {}
            )
        result["heads"][head] = item
    return result
