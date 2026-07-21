"""Payload-safe v0.2 diagnostics and descriptive cross-version comparison."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from promptsec.policybench.io import read_json_object, write_json
from promptsec.training.labels import HEADS, MULTILABEL_HEADS, LabelMapping


def class_and_multilabel_audit(
    records: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, LabelMapping],
    *,
    split_name: str = "train",
) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for head in HEADS:
        mapping = mappings[head]
        parent = "derived" if head == "prompt_injection_verdict" else "annotations"
        if mapping.multilabel:
            positives = Counter(label for record in records for label in record[parent][head])
            all_zero = sum(not record[parent][head] for record in records)
            heads[head] = {
                "positive_support": {label: positives[label] for label in mapping.labels},
                "negative_support": {
                    label: len(records) - positives[label] for label in mapping.labels
                },
                "all_zero_vectors": all_zero,
                f"labels_absent_from_{split_name}": [
                    label for label in mapping.labels if positives[label] == 0
                ],
            }
        else:
            counts = Counter(record[parent][head] for record in records)
            heads[head] = {"class_distribution": dict(sorted(counts.items()))}
    return {
        "split": split_name,
        "records": len(records),
        "heads": heads,
        "payloads_included": False,
    }


def category_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(record["extensions"]["policybench_v0_1"]["blueprint"]["category"])
                for record in records
            ).items()
        )
    )


def v01_diagnostic_audit(report_root: str | Path | None) -> dict[str, Any]:
    expected = (
        "training_history.json",
        "validation_metrics.json",
        "test_metrics.json",
        "counterfactual_results.json",
        "hard_negative_results.json",
        "language_results.json",
        "multilabel_thresholds.json",
        "checkpoint_inventory.json",
        "resource_usage.json",
        "model_card.md",
        "run_manifest.json",
    )
    if report_root is None:
        return {
            "status": "NOT_AVAILABLE_LOCALLY",
            "configurable_in_colab": True,
            "expected_files": list(expected),
            "weights_required": False,
        }
    root = Path(report_root)
    available = [name for name in expected if (root / name).is_file()]
    machine = {}
    for name in available:
        if name.endswith(".json"):
            try:
                machine[name] = read_json_object(root / name)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                machine[name] = {"read_error": str(error)}
    return {
        "status": "AVAILABLE" if available else "NOT_FOUND",
        "report_root": str(root),
        "available_files": available,
        "missing_files": [name for name in expected if name not in available],
        "machine_readable_reports": machine,
        "payloads_included": False,
        "weights_required": False,
    }


def logical_consistency_results(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    mappings: Mapping[str, LabelMapping],
) -> dict[str, Any]:
    predicted = {
        head: predictions["logits"][head].argmax(axis=1)
        for head in HEADS
        if head not in MULTILABEL_HEADS
    }
    verdict_mapping = mappings["prompt_injection_verdict"]
    direct = predicted["prompt_injection_verdict"]
    truth = np.asarray(predictions["truth"]["prompt_injection_verdict"])
    components = (
        "instruction_presence",
        "instruction_presentation",
        "instruction_addressee",
        "authority_status",
    )
    components_correct = np.ones(len(truth), dtype=bool)
    for head in components:
        components_correct &= predicted[head] == np.asarray(predictions["truth"][head])
    verdict_correct = direct == truth
    detected = verdict_mapping.label_to_id["DETECTED"]
    verdict_logits = predictions["logits"]["prompt_injection_verdict"]
    shifted = verdict_logits - verdict_logits.max(axis=1, keepdims=True)
    detected_probability = np.exp(shifted)[:, detected] / np.exp(shifted).sum(axis=1)
    rows = []
    for index, record in enumerate(records):
        annotation = record["annotations"]
        rows.append(
            {
                "id": record["id"],
                "category": record["extensions"]["policybench_v0_1"]["blueprint"]["category"],
                "direct_correct": bool(verdict_correct[index]),
                "components_correct": bool(components_correct[index]),
                "false_positive": bool(truth[index] != detected and direct[index] == detected),
                "false_negative": bool(truth[index] == detected and direct[index] != detected),
                "detected_probability": float(detected_probability[index]),
                "goal_alignment": annotation["user_goal_alignment"],
                "policy_alignment": annotation["protected_policy_alignment"],
                "authority_status": annotation["authority_status"],
                "presentation": annotation["instruction_presentation"],
                "addressee": annotation["instruction_addressee"],
            }
        )
    false_positive = [row for row in rows if row["false_positive"]]
    conditional: dict[str, list[float]] = {}
    for row in rows:
        key = " + ".join(
            (
                row["goal_alignment"],
                row["policy_alignment"],
                row["authority_status"],
                row["presentation"],
                row["addressee"],
            )
        )
        conditional.setdefault(key, []).append(row["detected_probability"])
    return {
        "records": len(rows),
        "component_heads_correct_but_verdict_incorrect": sum(
            row["components_correct"] and not row["direct_correct"] for row in rows
        ),
        "verdict_correct_but_component_heads_inconsistent": sum(
            row["direct_correct"] and not row["components_correct"] for row in rows
        ),
        "false_positives_policy_conflict_alone": sum(
            row["policy_alignment"] == "CONFLICTING"
            and row["authority_status"] == "WITHIN_AUTHORITY"
            for row in false_positive
        ),
        "false_positives_goal_misalignment_alone": sum(
            row["goal_alignment"] == "MISALIGNED" and row["authority_status"] == "WITHIN_AUTHORITY"
            for row in false_positive
        ),
        "false_negatives_outside_or_spoofed": sum(
            row["false_negative"] and row["authority_status"] in {"OUTSIDE_AUTHORITY", "SPOOFED"}
            for row in rows
        ),
        "conditional_detected_probability": {
            key: {"records": len(values), "mean_probability": float(np.mean(values))}
            for key, values in sorted(conditional.items())
        },
        "diagnostic_rows": rows,
        "payloads_included": False,
        "scope_warning": "The frozen verdict is not a complete model of real-world security risk.",
    }


def write_cross_version_comparison(
    destination: str | Path,
    *,
    v01_audit: Mapping[str, Any],
    v02_summary: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    v01_machine = v01_audit.get("machine_readable_reports", {})
    v01_metrics = _flatten_comparison_metrics(
        validation=v01_machine.get("validation_metrics.json", {}),
        tests=v01_machine.get("test_metrics.json", {}),
        counterfactual=v01_machine.get("counterfactual_results.json", {}),
        hard_negative=v01_machine.get("hard_negative_results.json", {}),
        resources=v01_machine.get("resource_usage.json", {}),
    )
    v02_metrics = _flatten_comparison_metrics(
        validation=v02_summary.get("validation_metrics", {}),
        tests=v02_summary.get("test_metrics", {}),
        counterfactual=v02_summary.get("counterfactual_results", {}),
        hard_negative=v02_summary.get("hard_negative_results", {}),
        resources=v02_summary.get("resource_usage", {}),
    )
    metric_rows = []
    for metric in sorted(set(v01_metrics) | set(v02_metrics)):
        left = v01_metrics.get(metric)
        right = v02_metrics.get(metric)
        metric_rows.append(
            {
                "metric": metric,
                "v0_1": left,
                "v0_2": right,
                "v0_2_minus_v0_1": (
                    right - left if left is not None and right is not None else None
                ),
            }
        )
    result = {
        "status": (
            "COMPARABLE" if v01_audit.get("status") == "AVAILABLE" else "V0_1_REPORTS_MISSING"
        ),
        "v0_1": dict(v01_audit),
        "v0_2": dict(v02_summary),
        "metric_comparison": metric_rows,
        "selection_used_test_metrics": False,
        "bootstrap_note": (
            "Use counterfactual groups for pair metrics, family groups when available, "
            "and records otherwise. Do not infer superiority from a small point estimate."
        ),
        "paired_uncertainty": {
            "status": "UNAVAILABLE_FROM_AGGREGATE_V0_1_REPORTS",
            "reason": (
                "Confidence intervals require paired record/group statistics; aggregate "
                "reports are not silently treated as paired observations."
            ),
        },
    }
    write_json(root / "v0_1_vs_v0_2.json", result)
    fields = ("metric", "v0_1", "v0_2", "v0_2_minus_v0_1")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(metric_rows)
    (root / "v0_1_vs_v0_2.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (root / "v0_1_vs_v0_2.md").write_text(
        "# Comparaison PromptSec-FM v0.1 / v0.2\n\n"
        f"Statut : **{result['status']}**.\n\n"
        "- Sensibilité contrefactuelle améliorée : à établir depuis les rapports appariés.\n"
        "- Invariance dégradée : à établir depuis les groupes contrefactuels.\n"
        "- Sur-défense réduite : à établir par catégorie sur les mêmes splits.\n"
        "- Macro-F1 verdict modifiée : à établir avec incertitude.\n"
        "- F1 multilabel améliorée : à établir par label et globalement.\n"
        "- Français OOD modifié : descriptif seulement; le split diffère au-delà de la langue.\n"
        "- Coût temps/stockage : voir les rapports de ressources et pruning.\n"
        "- Modes d’échec restants : ne jamais masquer les résultats négatifs ou incertains.\n\n"
        "Aucune supériorité n’est déduite d’un faible écart ponctuel. Les intervalles bootstrap "
        "requièrent les unités appariées et ne sont pas inventés depuis des agrégats.\n",
        encoding="utf-8",
    )
    return result


def _flatten_comparison_metrics(
    *,
    validation: Mapping[str, Any],
    tests: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    hard_negative: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for split, metrics in {"validation": validation, **tests}.items():
        core = metrics.get("core_macro_f1") if isinstance(metrics, Mapping) else None
        if isinstance(core, (int, float)):
            output[f"{split}/core_macro_f1"] = float(core)
        if isinstance(metrics, Mapping):
            for head in HEADS:
                value = metrics.get(head, {})
                macro = value.get("macro_f1") if isinstance(value, Mapping) else None
                if isinstance(macro, (int, float)):
                    output[f"{split}/{head}/macro_f1"] = float(macro)
                if head == "prompt_injection_verdict" and isinstance(value, Mapping):
                    diagnostics = value.get("verdict_diagnostics", {})
                    for calibration_metric in (
                        "brier_score",
                        "expected_calibration_error_10_bins",
                    ):
                        calibration_value = diagnostics.get(calibration_metric)
                        if isinstance(calibration_value, (int, float)):
                            output[f"{split}/{head}/{calibration_metric}"] = float(
                                calibration_value
                            )
    for head in HEADS:
        values = counterfactual.get(head, {})
        if not isinstance(values, Mapping):
            continue
        for metric in (
            "expected_change_sensitivity",
            "invariant_prediction_consistency",
            "exact_group_accuracy",
        ):
            value = values.get(metric)
            if isinstance(value, (int, float)):
                output[f"test_counterfactual/{head}/{metric}"] = float(value)
    for split, split_values in hard_negative.items():
        if not isinstance(split_values, Mapping):
            continue
        grouped = split_values.get("by_language_and_category", {})
        for language, categories in grouped.items():
            for category, values in categories.items():
                rate = values.get("false_positive_rate")
                if isinstance(rate, (int, float)):
                    output[f"{split}/hard_negative/{language}/{category}/false_positive_rate"] = (
                        float(rate)
                    )
    for metric in (
        "duration_seconds",
        "peak_gpu_memory_bytes",
        "checkpoint_storage_bytes_before_retention",
    ):
        value = resources.get(metric)
        if isinstance(value, (int, float)):
            output[f"resources/{metric}"] = float(value)
    return output
