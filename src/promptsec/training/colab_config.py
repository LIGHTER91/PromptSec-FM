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


@dataclass(slots=True)
class TrainingSettings:
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

    def validate(self) -> None:
        if self.training_mode not in {SCIENTIFIC_EVALUATION, FINAL_SILVER_MODEL}:
            raise ValueError(f"unknown training mode: {self.training_mode}")
        if self.training_mode == FINAL_SILVER_MODEL and not self.final_silver_splits:
            raise ValueError("FINAL_SILVER_MODEL requires explicit final_silver_splits")
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_training_config(path: str | Path) -> TrainingSettings:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load training config {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != "0.1":
        raise ValueError("training config schema_version must be '0.1'")
    settings = TrainingSettings(
        **{key: item for key, item in value.items() if key != "schema_version"}
    )
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
