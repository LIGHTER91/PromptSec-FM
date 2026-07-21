"""Configuration and GPU preflight for Colab XLM-R training."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil
import yaml

SCIENTIFIC_EVALUATION = "SCIENTIFIC_EVALUATION"
FINAL_SILVER_MODEL = "FINAL_SILVER_MODEL"
BILINGUAL_FINAL_SILVER_MODEL = "BILINGUAL_FINAL_SILVER_MODEL"

_V01_FIELDS = (
    "model_name",
    "training_mode",
    "max_length",
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "dropout",
    "max_grad_norm",
    "early_stopping_patience",
    "seed",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "save_steps",
    "num_workers",
    "gradient_checkpointing",
    "global_multilabel_threshold",
    "per_label_thresholds",
    "optimize_multilabel_thresholds_on_validation",
    "head_tail_ratio",
    "minimum_single_class_weight",
    "maximum_single_class_weight",
    "minimum_pos_weight",
    "maximum_pos_weight",
    "head_weights",
    "final_silver_splits",
    "allow_cpu_smoke_test",
    "retry_cuda_oom_once",
    "hub_export_enabled",
)


@dataclass(slots=True)
class TrainingSettings:
    schema_version: str = "0.1"
    model_name: str = "FacebookAI/xlm-roberta-base"
    training_mode: str = SCIENTIFIC_EVALUATION
    max_length: int = 512
    epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    dropout: float = 0.10
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    seed: int = 20260718
    per_device_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    effective_batch_size: int = 16
    save_steps: int = 250
    num_workers: int = 0
    gradient_checkpointing: bool = True
    global_multilabel_threshold: float = 0.5
    per_label_thresholds: bool = False
    optimize_multilabel_thresholds_on_validation: bool = False
    head_tail_ratio: float = 0.75
    minimum_single_class_weight: float = 0.25
    maximum_single_class_weight: float = 4.0
    minimum_pos_weight: float = 0.25
    maximum_pos_weight: float = 10.0
    head_weights: dict[str, float] = field(default_factory=dict)
    final_silver_splits: list[str] = field(default_factory=list)
    allow_cpu_smoke_test: bool = True
    retry_cuda_oom_once: bool = True
    hub_export_enabled: bool = False
    experiment_name: str = "v0.1"
    use_category_balancing: bool = False
    maximum_category_oversampling_multiplier: float = 3.0
    use_pair_aware_sampler: bool = False
    counterfactual_batch_fraction: float = 0.50
    counterfactual_loss_weight: float = 0.0
    invariant_loss_weight: float = 1.0
    expected_change_loss_weight: float = 1.0
    counterfactual_margin: float = 0.10
    verdict_consistency_loss_weight: float = 0.0
    verdict_decoding_strategy: str = "DIRECT_HEAD"
    calibrate_hybrid_alpha_on_validation: bool = False
    hybrid_alpha_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    hybrid_false_positive_penalty: float = 0.25
    validation_selection_metric: str = "ORIGINAL_CORE_MACRO_F1"
    single_label_loss_mode: str = "WEIGHTED_CROSS_ENTROPY"
    focal_gamma: float = 2.0
    multilabel_loss_mode: str = "WEIGHTED_BCE"
    gamma_positive: float = 1.0
    gamma_negative: float = 4.0
    probability_clip: float = 0.05
    per_label_threshold_minimum_support: int = 2
    keep_best_checkpoint: bool = True
    keep_last_n_complete_checkpoints: int = 2
    prune_after_verified_best_export: bool = True
    checkpoint_pruning_dry_run: bool = True
    require_source_commit_match: bool = False
    v0_1_report_root: str | None = None

    def validate(self) -> None:
        if self.schema_version not in {"0.1", "0.2"}:
            raise ValueError("training config schema_version must be '0.1' or '0.2'")
        if self.training_mode not in {
            SCIENTIFIC_EVALUATION,
            FINAL_SILVER_MODEL,
            BILINGUAL_FINAL_SILVER_MODEL,
        }:
            raise ValueError(f"unknown training mode: {self.training_mode}")
        if (
            self.training_mode in {FINAL_SILVER_MODEL, BILINGUAL_FINAL_SILVER_MODEL}
            and not self.final_silver_splits
        ):
            raise ValueError(f"{self.training_mode} requires explicit final_silver_splits")
        if self.max_length < 32 or self.epochs < 1 or self.seed < 0:
            raise ValueError("invalid max_length, epochs, or seed")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("learning_rate, weight_decay, or warmup_ratio is invalid")
        if self.early_stopping_patience < 0 or self.max_grad_norm <= 0:
            raise ValueError("early_stopping_patience or max_grad_norm is invalid")
        if self.per_device_batch_size is not None and self.per_device_batch_size < 1:
            raise ValueError("per_device_batch_size must be positive")
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 < self.head_tail_ratio <= 1:
            raise ValueError("head_tail_ratio must be within (0, 1]")
        if not 0 < self.global_multilabel_threshold < 1:
            raise ValueError("global_multilabel_threshold must be within (0, 1)")
        if self.hub_export_enabled:
            raise ValueError("automatic Hugging Face Hub export is forbidden")
        if not 0 <= self.counterfactual_batch_fraction <= 1:
            raise ValueError("counterfactual_batch_fraction must be within [0, 1]")
        if self.maximum_category_oversampling_multiplier < 1:
            raise ValueError("maximum_category_oversampling_multiplier must be at least 1")
        if (
            min(
                self.counterfactual_loss_weight,
                self.invariant_loss_weight,
                self.expected_change_loss_weight,
                self.verdict_consistency_loss_weight,
                self.counterfactual_margin,
            )
            < 0
        ):
            raise ValueError("v0.2 auxiliary loss settings must be non-negative")
        if self.single_label_loss_mode not in {"WEIGHTED_CROSS_ENTROPY", "FOCAL_CROSS_ENTROPY"}:
            raise ValueError("unknown single_label_loss_mode")
        if self.multilabel_loss_mode not in {"WEIGHTED_BCE", "ASYMMETRIC_FOCAL_BCE"}:
            raise ValueError("unknown multilabel_loss_mode")
        if self.verdict_decoding_strategy not in {
            "DIRECT_HEAD",
            "DERIVED_FROM_COMPONENT_HEADS",
            "VALIDATION_CALIBRATED_HYBRID",
        }:
            raise ValueError("unknown verdict_decoding_strategy")
        if self.use_category_balancing and self.single_label_loss_mode == "FOCAL_CROSS_ENTROPY":
            if (
                self.maximum_single_class_weight > 5
                or self.maximum_category_oversampling_multiplier > 3
            ):
                raise ValueError("unsafe combination of focal CE, class weights, and oversampling")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.schema_version == "0.1":
            return {name: values[name] for name in _V01_FIELDS}
        return values

    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_training_config(path: str | Path) -> TrainingSettings:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load training config {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") not in {"0.1", "0.2"}:
        raise ValueError("training config schema_version must be '0.1' or '0.2'")
    settings = TrainingSettings(**value)
    settings.validate()
    return settings


def resolve_batch_strategy(total_vram_gib: float, effective_batch_size: int = 16) -> dict[str, int]:
    if total_vram_gib >= 15:
        batch_size = 8
    elif total_vram_gib >= 10:
        batch_size = 4
    else:
        batch_size = 2
    accumulation = max(1, math.ceil(effective_batch_size / batch_size))
    return {
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
    }


def environment_preflight(drive_root: str | Path | None = None) -> dict[str, Any]:
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    total_vram = 0
    free_vram = 0
    gpu_name = None
    bf16 = False
    if cuda:
        free_vram, total_vram = torch.cuda.mem_get_info()
        gpu_name = torch.cuda.get_device_name(0)
        bf16 = bool(torch.cuda.is_bf16_supported())
    drive_space = None
    if drive_root is not None and Path(drive_root).exists():
        drive_space = shutil.disk_usage(drive_root).free
    memory = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "total_gpu_vram_gib": total_vram / (1024**3),
        "free_gpu_vram_gib": free_vram / (1024**3),
        "bfloat16_supported": bf16,
        "system_ram_available_gib": memory.available / (1024**3),
        "system_ram_total_gib": memory.total / (1024**3),
        "drive_space_available_gib": (drive_space / (1024**3) if drive_space is not None else None),
        "precision": "bf16" if bf16 else ("fp16" if cuda else "fp32"),
    }


def require_cuda_for_full_training(environment: dict[str, Any], *, smoke_test: bool) -> None:
    if environment["cuda_available"]:
        return
    if smoke_test:
        return
    raise RuntimeError(
        "CUDA GPU is required for full XLM-R training. In Colab select Runtime > Change "
        "runtime type > GPU, reconnect, and rerun the preflight cell."
    )
