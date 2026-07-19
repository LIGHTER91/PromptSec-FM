"""Verdict false positives by official split, language, and generation category."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from promptsec.training.labels import LabelMapping

HARD_NEGATIVE_CATEGORIES = (
    "NO_INSTRUCTION",
    "QUOTED_OR_REPORTED",
    "HYPOTHETICAL",
    "HARD_NEGATIVE_SPECIAL_CASES",
    "ALIGNED_BUT_POLICY_CONFLICTING",
    "MISALIGNED_NOT_POLICY_CONFLICTING",
)


def hard_negative_results(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    mapping: LabelMapping,
    *,
    split: str,
) -> dict[str, Any]:
    by_id = {metadata["id"]: index for index, metadata in enumerate(predictions["metadata"])}
    detected = mapping.labels.index("DETECTED")
    selected = predictions["logits"]["prompt_injection_verdict"].argmax(axis=1)
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        category = record["extensions"]["policybench_v0_1"]["blueprint"]["category"]
        if category not in HARD_NEGATIVE_CATEGORIES:
            continue
        if record["derived"]["prompt_injection_verdict"] == "DETECTED":
            continue
        language = record["content"]["language"]
        false_positive = bool(selected[by_id[record["id"]]] == detected)
        groups[(language, category)].append(false_positive)
    return {
        "split": split,
        "by_language_and_category": {
            language: {
                category: {
                    "records": len(values),
                    "false_positives": sum(values),
                    "false_positive_rate": sum(values) / len(values),
                }
                for (item_language, category), values in groups.items()
                if item_language == language
            }
            for language in sorted({key[0] for key in groups})
        },
        "diagnostics_use_redacted_ids_only": True,
    }
