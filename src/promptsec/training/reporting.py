"""Local JSON/CSV/Markdown reports and standalone best-model export."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import write_json

REQUIRED_REPORTS = (
    "run_manifest.json",
    "environment_report.json",
    "dataset_integrity_report.json",
    "split_audit.json",
    "training_history.json",
    "validation_metrics.json",
    "test_metrics.json",
    "all_head_metrics.csv",
    "confusion_matrices.json",
    "multilabel_thresholds.json",
    "counterfactual_results.json",
    "hard_negative_results.json",
    "language_results.json",
    "truncation_report.json",
    "resource_usage.json",
    "checkpoint_inventory.json",
    "final_report.md",
    "model_card.md",
)

V02_ADDITIONAL_REPORTS = (
    "training_pair_audit.json",
    "sampling_report.json",
    "class_weight_report.json",
    "per_class_metrics.json",
    "multilabel_prevalence.json",
    "counterfactual_loss_report.json",
    "verdict_consistency_report.json",
    "verdict_decoding_calibration.json",
    "logical_consistency_results.json",
    "calibration_results.json",
    "checkpoint_pruning_manifest.json",
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize_reports(
    reports: str | Path,
    *,
    run_manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    integrity: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> None:
    root = Path(reports)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run_manifest.json", run_manifest)
    write_json(root / "environment_report.json", environment)
    write_json(root / "dataset_integrity_report.json", integrity)
    write_json(root / "split_audit.json", split_audit)


def _metric_rows(
    validation: Mapping[str, Any], tests: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in {"validation": validation, **tests}.items():
        for head, values in metrics.items():
            if head in {"core_macro_f1", "per_head_loss", "stratified"}:
                continue
            row = {"split": split, "head": head}
            for key, value in values.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    row[key] = value
            rows.append(row)
    return rows


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def build_model_card(
    *,
    model_name: str,
    parameter_count: int,
    validation: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> str:
    return f"""# PromptSec-FM XLM-R Multi-task Model

Base encoder: `{model_name}`  
Parameters: {parameter_count:,}  
Taxonomy: PromptSec-FM v1.0  
Training truth status: synthetic `SILVER_VALIDATED`, not human Gold.

The model uses a shared multilingual encoder and nine classification heads. Seven heads are
single-label; attack families and attack objectives are multi-label. Input is full canonical
context with deterministic section-aware truncation. Span extraction is out of scope for this
version and is planned as a separate future phase.

Best validation core macro F1: {validation.get("core_macro_f1", "not available")}.

## Limitations

Good PolicyBench performance does not prove real-world prompt-injection robustness. The official
language-OOD comparison may differ in factors beyond language. Labels are automatically generated
and validated SILVER annotations, not adjudicated human truth.

## Configuration

```json
{json.dumps(training_config, indent=2, sort_keys=True)}
```
"""


def build_final_report(summary: Mapping[str, Any]) -> str:
    settings = summary["training_config"]
    splits = summary["split_counts"]
    return f"""# PromptSec-FM XLM-R Multi-task Training Report

## 1. Dataset status and SILVER warning

The run used 6,000 immutable `SILVER_VALIDATED` PolicyBench records. Human validation remains
`PENDING`, annotator confidence remains `0.0`, and automatic Gold records remain `0`.

**The model was trained on synthetic SILVER labels. The reported results are not human-Gold
performance. Good PolicyBench performance does not prove real-world robustness.**

## 2. Hardware and Colab environment

```json
{json.dumps(summary["environment"], indent=2, sort_keys=True)}
```

## 3. Effective training configuration

```json
{json.dumps(settings, indent=2, sort_keys=True)}
```

## 4. Official splits used

```json
{json.dumps(splits, indent=2, sort_keys=True)}
```

Only `train` was optimized in SCIENTIFIC_EVALUATION mode; validation controlled early stopping,
thresholds, and checkpoint selection. Test and human-review records were not trained on.

## 5. Label mappings

Nine frozen schema-derived mappings and hashes are stored in each checkpoint and best-model export.

## 6. Model architecture

`FacebookAI/xlm-roberta-base`, first contextual-token pooling, shared dropout, seven softmax heads,
and two sigmoid multi-label heads. Parameter count: {summary["parameter_count"]:,}.

## 7. Loss configuration

Single-label mode: `{settings.get("single_label_loss_mode", "WEIGHTED_CROSS_ENTROPY")}`.
Multi-label mode: `{settings.get("multilabel_loss_mode", "WEIGHTED_BCE")}`. Per-head losses are
normalized before their configured weighted mean. Class and positive weights derive only from
training records. Counterfactual auxiliary weight:
`{settings.get("counterfactual_loss_weight", 0.0)}`; verdict-consistency weight:
`{settings.get("verdict_consistency_loss_weight", 0.0)}`.

## 8. Token length and truncation

See `truncation_report.json`; the candidate receives the largest reserved budget and is never
silently removed.

## 9. Training history

See `training_history.json` for total and per-head losses by epoch.

## 10. Best validation checkpoint

`{summary["training_result"].get("best_checkpoint")}` selected by
`{settings.get("validation_selection_metric", "ORIGINAL_CORE_MACRO_F1")}` with validation loss as
tie-breaker. Both original core macro F1 and robust validation score remain in training history.

## 11. Validation metrics

Core macro F1: {summary["validation_metrics"].get("core_macro_f1")}.

## 12. Official test metrics

See `test_metrics.json`. Each official test was evaluated once after checkpoint selection.

## 13. Counterfactual results

See `counterfactual_results.json` for pairwise, sensitivity, invariance, exact-group, transition,
and type-specific metrics. Machine-readable lexical-baseline comparisons are descriptive only and
are included when the existing reports were available; the CPU benchmark was not rerun.

Verdict decoding and validation-only alpha selection are recorded in
`verdict_decoding_calibration.json`; per-label multilabel thresholds and provenance are recorded in
`multilabel_thresholds.json`.

## 14. Hard-negative analysis

See `hard_negative_results.json` for verdict false positives by split, language, and category.

## 15. English/French comparison

See `language_results.json`. Differences are descriptive and are not attributed solely to language.

## 16. Runtime and GPU memory

Runtime: {summary["training_result"].get("duration_seconds")} seconds. Peak allocated GPU memory:
{summary["training_result"].get("peak_gpu_memory_bytes")} bytes.

## 17. Resume and checkpoint history

```json
{json.dumps(summary["training_result"].get("resume_events", []), indent=2)}
```

## 18. OOM events

```json
{json.dumps(summary.get("oom_events", []), indent=2)}
```

## 19. Limitations

No span head is trained. Synthetic SILVER results are not human-Gold results or evidence of
production robustness. FINAL_SILVER_MODEL reuse invalidates independent evaluation and is disabled
by default.

## 20. Reproduction command

```bash
{summary["reproduction_command"]}
```
"""


def write_completed_reports(reports: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(reports)
    validation = summary["validation_metrics"]
    tests = summary["test_metrics"]
    write_json(root / "validation_metrics.json", validation)
    write_json(root / "test_metrics.json", tests)
    _write_metrics_csv(root / "all_head_metrics.csv", _metric_rows(validation, tests))
    write_json(
        root / "confusion_matrices.json",
        {
            split: {
                head: values["confusion_matrix"]
                for head, values in metrics.items()
                if isinstance(values, Mapping) and "confusion_matrix" in values
            }
            for split, metrics in {"validation": validation, **tests}.items()
        },
    )
    write_json(root / "multilabel_thresholds.json", summary["thresholds"])
    write_json(
        root / "per_class_metrics.json",
        {
            split: {
                head: values.get("per_class", {})
                for head, values in metrics.items()
                if isinstance(values, Mapping) and "per_class" in values
            }
            for split, metrics in {"validation": validation, **tests}.items()
        },
    )
    write_json(root / "counterfactual_results.json", summary["counterfactual_results"])
    if "training_pair_audit" in summary:
        write_json(root / "training_pair_audit.json", summary["training_pair_audit"])
    write_json(root / "sampling_report.json", summary.get("sampling_report", {}))
    write_json(
        root / "counterfactual_loss_report.json",
        summary.get("counterfactual_loss_report", {}),
    )
    write_json(
        root / "verdict_consistency_report.json",
        summary.get("verdict_consistency_report", {}),
    )
    write_json(
        root / "verdict_decoding_calibration.json",
        summary.get("verdict_decoding", {}),
    )
    write_json(
        root / "logical_consistency_results.json",
        summary.get("logical_consistency_results", {}),
    )
    write_json(
        root / "calibration_results.json",
        {
            "multilabel": summary["thresholds"],
            "verdict": summary.get("verdict_decoding", {}),
        },
    )
    if summary.get("lexical_baseline_comparison") is not None:
        write_json(
            root / "lexical_baseline_comparison.json",
            summary["lexical_baseline_comparison"],
        )
    write_json(root / "hard_negative_results.json", summary["hard_negative_results"])
    write_json(root / "language_results.json", summary["language_results"])
    write_json(root / "truncation_report.json", summary["truncation_report"])
    write_json(root / "resource_usage.json", summary["resource_usage"])
    model_card = build_model_card(
        model_name=summary["training_config"]["model_name"],
        parameter_count=summary["parameter_count"],
        validation=validation,
        training_config=summary["training_config"],
    )
    _atomic_text(root / "model_card.md", model_card)
    _atomic_text(root / "final_report.md", build_final_report(summary))
    write_json(root / "run_manifest.json", summary["run_manifest"])


def export_best_model(
    destination: str | Path,
    *,
    model: Any,
    tokenizer: Any,
    metadata: Mapping[str, Any],
    model_card: str,
) -> Path:
    final = Path(destination)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        for name, value in metadata.items():
            write_json(temporary / f"{name}.json", value)
        _atomic_text(temporary / "README.md", model_card)
        files = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        }
        write_json(temporary / "checksums.json", files)
        if final.exists():
            shutil.rmtree(final)
        os.rename(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final
