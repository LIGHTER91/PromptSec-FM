#!/usr/bin/env python3
"""Train or evaluate the PromptSec-FM XLM-R multi-task model on official splits."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _ratio(value: str) -> float:
    result = float(value)
    if not 0 <= result < 1:
        raise argparse.ArgumentTypeError("must be within [0, 1)")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--reset-smoke-test",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--experiment", choices=("balanced", "relational", "focal"))
    parser.add_argument("--v0-1-report-root", type=Path)
    parser.add_argument(
        "--prune-checkpoints",
        action="store_true",
        help="Apply checksum-gated v0.2 pruning after export (default is dry-run).",
    )
    parser.add_argument("--max-train-records", type=_positive)
    parser.add_argument("--max-validation-records", type=_positive)
    parser.add_argument("--epochs", type=_positive)
    parser.add_argument("--per-device-batch-size", type=_positive)
    parser.add_argument("--gradient-accumulation-steps", type=_positive)
    parser.add_argument("--max-length", type=_positive)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--weight-decay", type=_nonnegative_float)
    parser.add_argument("--warmup-ratio", type=_ratio)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument(
        "--training-mode",
        choices=(
            "SCIENTIFIC_EVALUATION",
            "FINAL_SILVER_MODEL",
            "BILINGUAL_FINAL_SILVER_MODEL",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _settings_with_overrides(settings: Any, args: argparse.Namespace) -> Any:
    values = {
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "early_stopping_patience": args.early_stopping_patience,
        "training_mode": args.training_mode,
    }
    return dataclasses.replace(
        settings, **{key: value for key, value in values.items() if value is not None}
    )


def _experiment_settings(settings: Any, experiment: str | None) -> Any:
    if experiment is None:
        return settings
    common = {
        "experiment_name": experiment,
        "use_category_balancing": True,
        "per_label_thresholds": True,
        "optimize_multilabel_thresholds_on_validation": True,
    }
    if experiment == "balanced":
        return dataclasses.replace(
            settings,
            **common,
            use_pair_aware_sampler=False,
            counterfactual_loss_weight=0.0,
            verdict_consistency_loss_weight=0.0,
            verdict_decoding_strategy="DIRECT_HEAD",
            calibrate_hybrid_alpha_on_validation=False,
            multilabel_loss_mode="WEIGHTED_BCE",
        )
    if experiment == "relational":
        return dataclasses.replace(
            settings,
            **common,
            use_pair_aware_sampler=True,
            counterfactual_loss_weight=max(0.25, settings.counterfactual_loss_weight),
            verdict_consistency_loss_weight=max(0.20, settings.verdict_consistency_loss_weight),
            verdict_decoding_strategy="VALIDATION_CALIBRATED_HYBRID",
            calibrate_hybrid_alpha_on_validation=True,
            multilabel_loss_mode="WEIGHTED_BCE",
        )
    return dataclasses.replace(
        _experiment_settings(settings, "relational"),
        experiment_name="focal",
        multilabel_loss_mode="ASYMMETRIC_FOCAL_BCE",
    )


def _lexical_baseline_comparison(
    test_metrics: dict[str, Any], counterfactual: dict[str, Any]
) -> dict[str, Any] | None:
    baseline_root = REPOSITORY_ROOT / "reports" / "cpu-baselines-v0.1"
    best_path = baseline_root / "best_models.json"
    counterfactual_path = baseline_root / "counterfactual_results.json"
    if not best_path.is_file() or not counterfactual_path.is_file():
        return None
    best = json.loads(best_path.read_text(encoding="utf-8"))
    baseline_counterfactual = json.loads(counterfactual_path.read_text(encoding="utf-8")).get(
        "experiments", []
    )
    comparison: dict[str, Any] = {
        "status": "DESCRIPTIVE_ONLY",
        "baseline_rerun": False,
        "warning": "These differences are descriptive and were not used for model selection.",
        "test_macro_f1": {},
        "counterfactual_relation_metrics": {},
    }
    for head, selection in best.items():
        comparison["test_macro_f1"][head] = {}
        for split, lexical_value in selection.get("test_macro_f1", {}).items():
            transformer_value = test_metrics.get(split, {}).get(head, {}).get("macro_f1")
            if transformer_value is None:
                continue
            comparison["test_macro_f1"][head][split] = {
                "transformer": transformer_value,
                "lexical_baseline": lexical_value,
                "transformer_minus_lexical": transformer_value - lexical_value,
            }
        selected = next(
            (
                item
                for item in baseline_counterfactual
                if item.get("target") == head
                and item.get("model_family") == selection.get("model_family")
                and item.get("ablation") == selection.get("ablation")
            ),
            None,
        )
        if selected is None or head not in counterfactual:
            continue
        comparison["counterfactual_relation_metrics"][head] = {}
        for metric in (
            "pairwise_accuracy",
            "expected_change_sensitivity",
            "invariant_prediction_consistency",
            "exact_group_accuracy",
        ):
            transformer_value = counterfactual[head].get(metric)
            lexical_value = selected.get(metric)
            comparison["counterfactual_relation_metrics"][head][metric] = {
                "transformer": transformer_value,
                "lexical_baseline": lexical_value,
                "transformer_minus_lexical": (
                    transformer_value - lexical_value
                    if transformer_value is not None and lexical_value is not None
                    else None
                ),
            }
    return comparison


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from promptsec.policybench.io import write_json
    from promptsec.training.checkpoints import checkpoint_inventory, verify_checkpoint
    from promptsec.training.colab_config import (
        BILINGUAL_FINAL_SILVER_MODEL,
        FINAL_SILVER_MODEL,
        environment_preflight,
        load_training_config,
        require_cuda_for_full_training,
        resolve_batch_strategy,
    )
    from promptsec.training.counterfactual_evaluation import (
        evaluate_counterfactual_predictions,
    )
    from promptsec.training.dataset import (
        EVALUATION_SPLITS,
        MultitaskCollator,
        PolicyBenchTorchDataset,
        load_training_dataset,
        summarize_truncation,
    )
    from promptsec.training.diagnostics_v02 import (
        class_and_multilabel_audit,
        logical_consistency_results,
        v01_diagnostic_audit,
        write_cross_version_comparison,
    )
    from promptsec.training.evaluation import (
        metrics_from_predictions,
        predict_dataloader,
        stratified_metrics,
    )
    from promptsec.training.hard_negative_evaluation import hard_negative_results
    from promptsec.training.labels import mappings_fingerprint
    from promptsec.training.language_evaluation import compare_language_metrics
    from promptsec.training.metrics import select_multilabel_thresholds
    from promptsec.training.multitask_model import PromptSecMultitaskModel
    from promptsec.training.pairing import audit_counterfactual_groups
    from promptsec.training.reporting import (
        build_model_card,
        export_best_model,
        initialize_reports,
        write_completed_reports,
    )
    from promptsec.training.serialization import SPECIAL_TOKENS
    from promptsec.training.trainer import MultitaskTrainer

    settings = _experiment_settings(
        _settings_with_overrides(load_training_config(args.config), args), args.experiment
    )
    if args.v0_1_report_root is not None:
        settings = dataclasses.replace(settings, v0_1_report_root=str(args.v0_1_report_root))
    if args.prune_checkpoints:
        settings = dataclasses.replace(settings, checkpoint_pruning_dry_run=False)
    if args.resume is None:
        args.resume = not (args.smoke_test and settings.schema_version == "0.2")
    if args.reset_smoke_test is None:
        args.reset_smoke_test = args.smoke_test and settings.schema_version == "0.2"
    if args.smoke_test:
        settings = dataclasses.replace(
            settings,
            epochs=args.epochs or 1,
            max_length=args.max_length or 128,
            # The epoch checkpoint is enough for the resume probe. A checkpoint
            # per optimizer step would duplicate several GiB of state on Drive.
            save_steps=0,
        )
        args.max_train_records = args.max_train_records or 32
        args.max_validation_records = args.max_validation_records or 16
        args.output = args.output / "smoke-test"
        args.reports = args.reports / "smoke-test"
        if args.reset_smoke_test:
            from promptsec.training.retention import reset_isolated_smoke_directories

            reset_isolated_smoke_directories((args.output, args.reports))
    settings.validate()
    bundle = load_training_dataset(args.dataset)
    final_modes = {FINAL_SILVER_MODEL, BILINGUAL_FINAL_SILVER_MODEL}
    if settings.training_mode in final_modes:
        selected = settings.final_silver_splits
        invalid = set(selected).difference(bundle.records_by_split)
        if invalid:
            raise ValueError(f"unknown final SILVER pool splits: {sorted(invalid)}")
        records_by_split = dict(bundle.records_by_split)
        records_by_split["train"] = [
            record for split in selected for record in bundle.records_by_split[split]
        ]
        bundle = dataclasses.replace(bundle, records_by_split=records_by_split)
    environment = environment_preflight(args.output.parent)
    source_commit_hash = _git_commit()
    environment["source_commit_hash"] = source_commit_hash
    require_cuda_for_full_training(environment, smoke_test=args.smoke_test)
    if settings.per_device_batch_size is None:
        strategy = resolve_batch_strategy(environment["total_gpu_vram_gib"])
        settings.per_device_batch_size = strategy["per_device_batch_size"]
    if settings.gradient_accumulation_steps is None:
        settings.gradient_accumulation_steps = max(
            1, -(-settings.effective_batch_size // settings.per_device_batch_size)
        )
    settings.validate()
    resolved_strategy = {
        "per_device_batch_size": settings.per_device_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "effective_batch_size": (
            settings.per_device_batch_size * settings.gradient_accumulation_steps
        ),
    }
    run_manifest = {
        "schema_version": settings.schema_version,
        "experiment_name": settings.experiment_name,
        "run_kind": "SMOKE_TEST" if args.smoke_test else settings.training_mode,
        "dataset": str(bundle.root),
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "split_hashes": bundle.split_hashes,
        "training_config_hash": settings.fingerprint(),
        "label_mapping_hash": mappings_fingerprint(bundle.mappings),
        "source_commit_hash": source_commit_hash,
        "resolved_batch_strategy": resolved_strategy,
        "selection_split": "validation",
        "selection_metric": settings.validation_selection_metric,
        "test_used_for_selection": False,
        "truth_status": "SYNTHETIC_SILVER_NOT_HUMAN_GOLD",
        "final_silver_warning": (
            "No independent evaluation remains for any split reused in FINAL_SILVER_MODEL."
            if settings.training_mode in final_modes
            else None
        ),
        "full_local_training_permitted": False,
    }
    initialize_reports(
        args.reports,
        run_manifest=run_manifest,
        environment=environment,
        integrity=bundle.integrity_report,
        split_audit=bundle.split_audit,
    )
    pair_audit = audit_counterfactual_groups(bundle.records_by_split)
    write_json(args.reports / "training_pair_audit.json", pair_audit)
    prevalence = {
        split: class_and_multilabel_audit(
            bundle.records_by_split[split], bundle.mappings, split_name=split
        )
        for split in ("train", "validation")
    }
    write_json(args.reports / "multilabel_prevalence.json", prevalence)
    v01_audit = v01_diagnostic_audit(settings.v0_1_report_root)
    write_json(args.reports / "v0_1_diagnostic_audit.json", v01_audit)
    write_json(args.reports / "effective_training_config.json", settings.as_dict())
    write_json(args.reports / "checkpoint_inventory.json", checkpoint_inventory(args.output))
    tokenizer_source: str | Path = settings.model_name
    model_source: str | Path | None = None
    if args.checkpoint is not None:
        verify_checkpoint(args.checkpoint)
        tokenizer_source = args.checkpoint / "tokenizer"
        model_source = args.checkpoint / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    mapping_data = {head: mapping.to_dict() for head, mapping in bundle.mappings.items()}
    if model_source is None:
        model = PromptSecMultitaskModel.from_encoder_pretrained(
            settings.model_name,
            head_dimensions={
                head: len(mapping.labels) for head, mapping in bundle.mappings.items()
            },
            label_mappings=mapping_data,
            special_tokens=list(SPECIAL_TOKENS),
            dropout=settings.dropout,
        )
        model.resize_encoder_token_embeddings(len(tokenizer))
    else:
        model = PromptSecMultitaskModel.from_pretrained(model_source)
    if settings.schema_version == "0.1":
        model.config.max_length = settings.max_length
    else:
        model.config.promptsec_max_length = settings.max_length
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = MultitaskTrainer(
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        settings=settings,
        output=args.output,
        reports=args.reports,
        environment=environment,
        resume=bool(args.resume),
        smoke_test=args.smoke_test,
        max_train_records=args.max_train_records,
        max_validation_records=args.max_validation_records,
    )
    write_json(
        args.reports / "class_weight_report.json",
        {
            "single_label": {
                head: dict(zip(bundle.mappings[head].labels, values, strict=True))
                for head, values in trainer.class_weights.items()
            },
            "multilabel_pos_weight": {
                head: dict(zip(bundle.mappings[head].labels, values, strict=True))
                for head, values in trainer.positive_weights.items()
            },
            "training_records_only": True,
        },
    )
    oom_events: list[dict[str, Any]] = []
    if args.evaluate_only:
        if args.checkpoint is None:
            raise ValueError("--evaluate-only requires --checkpoint")
        training_result = {
            "best_checkpoint": str(args.checkpoint),
            "duration_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
            "resume_events": [],
        }
        write_json(args.reports / "training_history.json", {"epochs": [], "evaluate_only": True})
    else:
        try:
            training_result = trainer.train()
        except torch.cuda.OutOfMemoryError as error:
            if not settings.retry_cuda_oom_once or settings.per_device_batch_size == 1:
                raise
            event = {
                "status": "RECOVERED_RETRY_SCHEDULED",
                "error": type(error).__name__,
                "previous_batch_size": settings.per_device_batch_size,
            }
            settings.per_device_batch_size = max(1, settings.per_device_batch_size // 2)
            settings.gradient_accumulation_steps *= 2
            event["retry_batch_size"] = settings.per_device_batch_size
            event["retry_gradient_accumulation_steps"] = settings.gradient_accumulation_steps
            oom_events.append(event)
            write_json(args.reports / "failed_oom_attempt.json", event)
            torch.cuda.empty_cache()
            trainer.settings = settings
            trainer.resume = True
            training_result = trainer.train()
    best_checkpoint = Path(training_result["best_checkpoint"])
    trainer._load_model_weights(best_checkpoint)
    if args.smoke_test:
        training_result["resume_probe"] = trainer.verify_resume_probe(best_checkpoint)
        trainer._load_model_weights(best_checkpoint)

    device = trainer.device
    collator = MultitaskCollator(tokenizer)

    def predict_split(split: str, limit: int | None = None) -> tuple[Any, Any]:
        records = bundle.records_by_split[split][:limit]
        dataset = PolicyBenchTorchDataset(
            records,
            tokenizer,
            bundle.mappings,
            max_length=settings.max_length,
            head_tail_ratio=settings.head_tail_ratio,
        )
        loader = DataLoader(
            dataset,
            batch_size=settings.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=environment["cuda_available"],
        )
        return predict_dataloader(model, loader, device), dataset

    validation_predictions, validation_dataset = predict_split(
        "validation", args.max_validation_records if args.smoke_test else None
    )
    direct_predictions_by_split: dict[str, Any] = {"validation": validation_predictions}
    thresholds: dict[str, Any] = {
        "methodology": "fixed global 0.5 selected before test evaluation",
        "selection_split": "validation",
        "attack_families": settings.global_multilabel_threshold,
        "attack_objectives": settings.global_multilabel_threshold,
    }
    if settings.optimize_multilabel_thresholds_on_validation:
        import numpy as np

        from promptsec.training.calibration import calibrate_per_label_thresholds

        for head in ("attack_families", "attack_objectives"):
            values = validation_predictions["logits"][head]
            probabilities = 1 / (1 + np.exp(-values))
            if settings.schema_version == "0.2" and settings.per_label_thresholds:
                calibrated = calibrate_per_label_thresholds(
                    np.asarray(validation_predictions["truth"][head]),
                    probabilities,
                    bundle.mappings[head].labels,
                    minimum_positive_support=settings.per_label_threshold_minimum_support,
                    minimum_negative_support=settings.per_label_threshold_minimum_support,
                )
                thresholds[head] = calibrated["thresholds"]
                thresholds[f"{head}_provenance"] = calibrated
            else:
                thresholds[head] = select_multilabel_thresholds(
                    np.asarray(validation_predictions["truth"][head]),
                    probabilities,
                    per_label=settings.per_label_thresholds,
                )
        thresholds["methodology"] = "validation-only deterministic F1 calibration"
        thresholds["test_metrics_used"] = False
    metric_thresholds = {
        head: thresholds[head] for head in ("attack_families", "attack_objectives")
    }
    verdict_calibration: dict[str, Any] = {
        "selected_strategy": settings.verdict_decoding_strategy,
        "selected_alpha": 1.0,
        "selection_split": "validation",
        "test_metrics_used": False,
    }
    direct_validation_metrics = metrics_from_predictions(
        validation_predictions, bundle.mappings, metric_thresholds
    )
    if settings.schema_version == "0.2":
        import copy

        import numpy as np

        from promptsec.training.verdict import (
            VALIDATION_CALIBRATED_HYBRID,
            decode_verdict_probabilities,
            select_hybrid_alpha,
        )

        def verdict_probabilities(raw: dict[str, Any], strategy: str, alpha: float) -> Any:
            tensors = {
                head: torch.as_tensor(values, dtype=torch.float32)
                for head, values in raw["logits"].items()
            }
            return decode_verdict_probabilities(
                tensors,
                bundle.mappings,
                strategy=strategy,
                alpha=alpha,
            )

        direct_derived = verdict_probabilities(
            validation_predictions, VALIDATION_CALIBRATED_HYBRID, 0.5
        )
        if settings.calibrate_hybrid_alpha_on_validation:
            verdict_calibration = select_hybrid_alpha(
                validation_predictions["truth"]["prompt_injection_verdict"],
                direct_derived["direct"].numpy(),
                direct_derived["derived"].numpy(),
                [item["category"] for item in validation_predictions["metadata"]],
                bundle.mappings["prompt_injection_verdict"].labels,
                alpha_grid=settings.hybrid_alpha_grid,
                false_positive_penalty=settings.hybrid_false_positive_penalty,
            )
            verdict_calibration["selected_strategy"] = settings.verdict_decoding_strategy
        selected_alpha = float(verdict_calibration.get("selected_alpha", 1.0))
        final = verdict_probabilities(
            validation_predictions,
            settings.verdict_decoding_strategy,
            selected_alpha,
        )["final"].numpy()
        validation_predictions = copy.deepcopy(validation_predictions)
        validation_predictions["logits"]["prompt_injection_verdict"] = np.log(
            np.clip(final, 1e-12, 1.0)
        )
    validation_metrics = metrics_from_predictions(
        validation_predictions, bundle.mappings, metric_thresholds
    )
    validation_metrics["direct_head_secondary"] = direct_validation_metrics[
        "prompt_injection_verdict"
    ]
    validation_metrics["stratified"] = {
        field: stratified_metrics(validation_predictions, bundle.mappings, metric_thresholds, field)
        for field in ("language", "domain", "category")
    }
    test_metrics: dict[str, Any] = {}
    raw_predictions: dict[str, Any] = {"validation": validation_predictions}
    datasets: dict[str, Any] = {"validation": validation_dataset}
    if not args.smoke_test and settings.training_mode not in final_modes:
        for split in EVALUATION_SPLITS[1:]:
            raw, dataset = predict_split(split)
            if settings.schema_version == "0.2":
                import copy

                import numpy as np

                direct_raw = copy.deepcopy(raw)
                final = verdict_probabilities(
                    raw,
                    settings.verdict_decoding_strategy,
                    float(verdict_calibration.get("selected_alpha", 1.0)),
                )["final"].numpy()
                raw = copy.deepcopy(raw)
                raw["logits"]["prompt_injection_verdict"] = np.log(np.clip(final, 1e-12, 1.0))
                direct_predictions_by_split[split] = direct_raw
            else:
                direct_predictions_by_split[split] = raw
            raw_predictions[split] = raw
            datasets[split] = dataset
            metrics = metrics_from_predictions(raw, bundle.mappings, metric_thresholds)
            if settings.schema_version == "0.2":
                metrics["direct_head_secondary"] = metrics_from_predictions(
                    direct_raw, bundle.mappings, metric_thresholds
                )["prompt_injection_verdict"]
            metrics["stratified"] = {
                field: stratified_metrics(raw, bundle.mappings, metric_thresholds, field)
                for field in ("language", "domain", "category")
            }
            test_metrics[split] = metrics
    counterfactual = {}
    hard_negative = {}
    language = {}
    if "test_counterfactual" in raw_predictions:
        counterfactual = evaluate_counterfactual_predictions(
            bundle.records_by_split["test_counterfactual"],
            raw_predictions["test_counterfactual"],
            bundle.mappings,
            metric_thresholds,
        )
    lexical_comparison = _lexical_baseline_comparison(test_metrics, counterfactual)
    if lexical_comparison is not None:
        counterfactual["descriptive_lexical_baseline_comparison"] = lexical_comparison
    for split, raw in raw_predictions.items():
        records = bundle.records_by_split[split]
        if args.smoke_test:
            records = records[: args.max_validation_records]
        hard_negative[split] = hard_negative_results(
            records,
            raw,
            bundle.mappings["prompt_injection_verdict"],
            split=split,
        )
    if "test_language_ood" in test_metrics:
        language = compare_language_metrics(
            validation_metrics["stratified"]["language"]["en"],
            test_metrics["test_language_ood"]["stratified"]["language"]["fr"],
        )
    truncation = {
        split: summarize_truncation([dataset[index]["truncation"] for index in range(len(dataset))])
        for split, dataset in datasets.items()
    }
    logical_consistency = {
        split: logical_consistency_results(
            (
                bundle.records_by_split[split][
                    : args.max_validation_records if args.smoke_test else None
                ]
            ),
            raw,
            bundle.mappings,
        )
        for split, raw in direct_predictions_by_split.items()
    }
    training_config = settings.as_dict()
    training_config["resolved_batch_strategy"] = resolved_strategy
    training_config["special_tokens"] = list(SPECIAL_TOKENS)
    reproduction = (
        f"python scripts/train_xlmr_multitask.py --config {args.config} "
        f"--dataset {args.dataset} "
        f"--output {args.output.parent if args.smoke_test else args.output} "
        f"--reports {args.reports.parent if args.smoke_test else args.reports} --resume"
    )
    resource_usage = {
        "duration_seconds": training_result.get("duration_seconds"),
        "peak_gpu_memory_bytes": training_result.get("peak_gpu_memory_bytes"),
        "resolved_batch_strategy": resolved_strategy,
        "oom_events": oom_events,
        "checkpoint_storage_bytes_before_retention": checkpoint_inventory(args.output).get(
            "total_complete_size_bytes", 0
        ),
    }
    run_manifest.update(
        {
            "status": "COMPLETE",
            "parameter_count": parameter_count,
            "best_checkpoint": str(best_checkpoint),
            "thresholds": thresholds,
            "resume_events": training_result.get("resume_events", []),
            "oom_events": oom_events,
        }
    )
    summary = {
        "run_manifest": run_manifest,
        "environment": environment,
        "training_config": training_config,
        "split_counts": bundle.integrity_report["split_counts"],
        "parameter_count": parameter_count,
        "training_result": training_result,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "thresholds": thresholds,
        "verdict_decoding": verdict_calibration,
        "counterfactual_results": counterfactual,
        "logical_consistency_results": logical_consistency,
        "training_pair_audit": pair_audit,
        "sampling_report": training_result.get("sampling_report", {}),
        "counterfactual_loss_report": {
            "enabled": settings.counterfactual_loss_weight > 0,
            "weight": settings.counterfactual_loss_weight,
            "per_epoch": [
                item.get("training_head_losses", {}) for item in training_result.get("history", [])
            ],
        },
        "verdict_consistency_report": {
            "enabled": settings.verdict_consistency_loss_weight > 0,
            "weight": settings.verdict_consistency_loss_weight,
            "canonical_authority_labels": ["OUTSIDE_AUTHORITY", "SPOOFED"],
        },
        "lexical_baseline_comparison": lexical_comparison,
        "hard_negative_results": hard_negative,
        "language_results": language,
        "truncation_report": truncation,
        "resource_usage": resource_usage,
        "oom_events": oom_events,
        "reproduction_command": reproduction,
    }
    model_card = build_model_card(
        model_name=settings.model_name,
        parameter_count=parameter_count,
        validation=validation_metrics,
        training_config=training_config,
    )
    export_best_model(
        args.output / "best_model",
        model=model,
        tokenizer=tokenizer,
        metadata={
            "label_mappings": {
                "taxonomy_version": "1.0",
                "mapping_hash": mappings_fingerprint(bundle.mappings),
                "heads": mapping_data,
            },
            "classification_thresholds": thresholds,
            "verdict_decoding_configuration": verdict_calibration,
            "class_weights": {
                "single_label": trainer.class_weights,
                "multilabel_pos_weight": trainer.positive_weights,
            },
            "preprocessing_configuration": {
                "max_length": settings.max_length,
                "head_tail_ratio": settings.head_tail_ratio,
                "special_tokens": list(SPECIAL_TOKENS),
            },
            "dataset_fingerprint": {
                "manifest_sha256": bundle.manifest_sha256,
                "split_hashes": bundle.split_hashes,
            },
            "training_configuration": {
                **training_config,
                "training_config_hash": settings.fingerprint(),
                "source_commit_hash": source_commit_hash,
            },
            "validation_summary": validation_metrics,
        },
        model_card=model_card,
    )
    write_completed_reports(args.reports, summary)
    write_json(args.reports / "checkpoint_inventory.json", checkpoint_inventory(args.output))
    if settings.schema_version == "0.2":
        if args.evaluate_only:
            write_json(
                args.reports / "checkpoint_pruning_manifest.json",
                {"status": "NOT_APPLICABLE_EVALUATE_ONLY", "reclaimed_bytes": 0},
            )
        else:
            from promptsec.training.retention import (
                apply_checkpoint_retention,
                plan_checkpoint_retention,
            )

            plan = plan_checkpoint_retention(
                args.output,
                best_checkpoint=best_checkpoint,
                keep_last_n_complete=settings.keep_last_n_complete_checkpoints,
                compatibility=trainer.compatibility,
                run_kind=trainer.run_kind,
            )
            apply_checkpoint_retention(
                plan,
                verified_best_export=args.output / "best_model",
                dry_run=(
                    settings.checkpoint_pruning_dry_run
                    or not settings.prune_after_verified_best_export
                ),
                manifest_path=args.reports / "checkpoint_pruning_manifest.json",
            )
            write_json(
                args.reports / "checkpoint_inventory.json",
                checkpoint_inventory(args.output),
            )
        if not args.smoke_test:
            write_cross_version_comparison(
                args.reports.parent / "xlmr-base-multitask-v0.2-comparison",
                v01_audit=v01_audit,
                v02_summary={
                    "experiment_name": settings.experiment_name,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "counterfactual_results": counterfactual,
                    "hard_negative_results": hard_negative,
                    "language_results": language,
                    "resource_usage": resource_usage,
                    "checkpoint_inventory": checkpoint_inventory(args.output),
                },
            )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
