"""Deterministic custom Trainer with validation-only selection and safe resume."""

from __future__ import annotations

import contextlib
import gc
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from promptsec.policybench.io import write_json
from promptsec.training.checkpoints import (
    checkpoint_inventory,
    find_latest_compatible_checkpoint,
    load_training_state,
    save_checkpoint_atomic,
)
from promptsec.training.colab_config import TrainingSettings
from promptsec.training.dataset import (
    MultitaskCollator,
    PolicyBenchTorchDataset,
    TrainingDatasetBundle,
)
from promptsec.training.evaluation import metrics_from_predictions, predict_dataloader
from promptsec.training.labels import mappings_fingerprint
from promptsec.training.losses import compute_training_class_weights


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultitaskTrainer:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        bundle: TrainingDatasetBundle,
        settings: TrainingSettings,
        output: str | Path,
        reports: str | Path,
        environment: Mapping[str, Any],
        resume: bool,
        smoke_test: bool,
        max_train_records: int | None = None,
        max_validation_records: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.bundle = bundle
        self.settings = settings
        self.output = Path(output)
        self.reports = Path(reports)
        self.environment = dict(environment)
        self.resume = resume
        self.smoke_test = smoke_test
        self.run_kind = "SMOKE_TEST" if smoke_test else settings.training_mode
        self.output.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        train_records = bundle.records_by_split["train"][:max_train_records]
        validation_records = bundle.records_by_split["validation"][:max_validation_records]
        self.train_dataset = PolicyBenchTorchDataset(
            train_records,
            tokenizer,
            bundle.mappings,
            max_length=settings.max_length,
            head_tail_ratio=settings.head_tail_ratio,
        )
        self.validation_dataset = PolicyBenchTorchDataset(
            validation_records,
            tokenizer,
            bundle.mappings,
            max_length=settings.max_length,
            head_tail_ratio=settings.head_tail_ratio,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if settings.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        class_weights, positive_weights = compute_training_class_weights(
            train_records,
            bundle.mappings,
            single_clip=(
                settings.minimum_single_class_weight,
                settings.maximum_single_class_weight,
            ),
            positive_clip=(settings.minimum_pos_weight, settings.maximum_pos_weight),
        )
        self.model.set_loss_configuration(
            class_weights=class_weights,
            positive_weights=positive_weights,
            head_weights=settings.head_weights,
        )
        self.class_weights = {head: value.tolist() for head, value in class_weights.items()}
        self.positive_weights = {head: value.tolist() for head, value in positive_weights.items()}
        self.thresholds: dict[str, float] = {
            "attack_families": settings.global_multilabel_threshold,
            "attack_objectives": settings.global_multilabel_threshold,
        }
        self.compatibility = {
            "dataset_manifest_sha256": bundle.manifest_sha256,
            "split_hashes": bundle.split_hashes,
            "training_config_hash": settings.fingerprint(),
            "label_mapping_hash": mappings_fingerprint(bundle.mappings),
            "model_name": settings.model_name,
            "special_tokens": list(self.model.config.special_tokens),
        }
        self.history: list[dict[str, Any]] = []
        self.resume_events: list[dict[str, Any]] = []

    def _dataloaders(self, train_epoch: int = 0) -> tuple[DataLoader, DataLoader]:
        collator = MultitaskCollator(self.tokenizer)
        generator = torch.Generator().manual_seed(self.settings.seed + train_epoch)
        train = DataLoader(
            self.train_dataset,
            batch_size=int(self.settings.per_device_batch_size),
            shuffle=True,
            collate_fn=collator,
            num_workers=self.settings.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
        validation = DataLoader(
            self.validation_dataset,
            batch_size=int(self.settings.per_device_batch_size),
            shuffle=False,
            collate_fn=collator,
            num_workers=self.settings.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train, validation

    def _optimizer_scheduler(self, train_batches: int) -> tuple[Any, Any, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        updates_per_epoch = math.ceil(
            train_batches / int(self.settings.gradient_accumulation_steps)
        )
        total_steps = updates_per_epoch * self.settings.epochs
        warmup = round(total_steps * self.settings.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=torch.cuda.is_available() and self.environment["precision"] == "fp16",
        )
        return optimizer, scheduler, scaler

    def _autocast(self) -> Any:
        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.environment["precision"] == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _load_model_weights(self, checkpoint: Path) -> None:
        safe_path = checkpoint / "model" / "model.safetensors"
        binary_path = checkpoint / "model" / "pytorch_model.bin"
        if safe_path.is_file():
            from safetensors.torch import load_file

            state = load_file(safe_path, device="cpu")
        elif binary_path.is_file():
            state = torch.load(binary_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError("checkpoint model weights are missing")
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _save_checkpoint(
        self,
        global_step: int,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        state: Mapping[str, Any],
        suffix: str | None = None,
    ) -> Path:
        name = f"checkpoint-{global_step:08d}"
        if suffix:
            name = f"{name}-{suffix}"
        destination = self.output / name
        if destination.exists():
            return destination
        mappings = {
            "taxonomy_version": "1.0",
            "mapping_hash": mappings_fingerprint(self.bundle.mappings),
            "heads": {head: mapping.to_dict() for head, mapping in self.bundle.mappings.items()},
        }
        return save_checkpoint_atomic(
            destination,
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            trainer_state=state,
            label_mappings=mappings,
            training_config={
                **self.settings.as_dict(),
                "class_weights": self.class_weights,
                "positive_weights": self.positive_weights,
                "thresholds": self.thresholds,
            },
            compatibility=self.compatibility,
            environment=self.environment,
            run_kind=self.run_kind,
        )

    def train(self) -> dict[str, Any]:
        set_deterministic_seed(self.settings.seed)
        train_loader, validation_loader = self._dataloaders()
        optimizer, scheduler, scaler = self._optimizer_scheduler(len(train_loader))
        start_epoch = 0
        start_batch = 0
        global_step = 0
        best_score = -math.inf
        best_loss = math.inf
        best_checkpoint = None
        patience = 0
        resumed_epoch_loss: list[float] = []
        resumed_head_losses: dict[str, list[float]] = {}
        latest = find_latest_compatible_checkpoint(
            self.output,
            compatibility=self.compatibility,
            run_kind=self.run_kind,
        )
        if self.resume:
            if latest is None and any(self.output.glob("checkpoint-*")):
                raise RuntimeError("RESUME=True but no compatible complete checkpoint was found")
            if latest is not None:
                self._load_model_weights(latest)
                state = load_training_state(
                    latest,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                start_epoch = int(state["epoch"])
                start_batch = int(state.get("batch_in_epoch", 0))
                global_step = int(state["global_step"])
                best_score = float(state["best_validation_score"])
                best_loss = float(state["best_validation_loss"])
                best_checkpoint = state.get("best_checkpoint")
                patience = int(state.get("early_stopping_counter", 0))
                self.history = list(state.get("history", []))
                resumed_epoch_loss = list(state.get("epoch_loss", []))
                resumed_head_losses = {
                    head: list(values)
                    for head, values in state.get("epoch_head_losses", {}).items()
                }
                self.resume_events.append(
                    {"checkpoint": str(latest), "global_step": global_step, "status": "RESTORED"}
                )
        elif latest is not None:
            raise RuntimeError("checkpoints exist; use --resume or a new output directory")
        train_loader, validation_loader = self._dataloaders(start_epoch)
        started = time.perf_counter()
        peak_gpu = 0
        for epoch in range(start_epoch, self.settings.epochs):
            if epoch != start_epoch:
                train_loader, _ = self._dataloaders(epoch)
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            head_losses: dict[str, list[float]] = defaultdict(list)
            epoch_loss = list(resumed_epoch_loss) if epoch == start_epoch else []
            if epoch == start_epoch:
                head_losses.update(resumed_head_losses)
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                labels = {head: value.to(self.device) for head, value in batch["labels"].items()}
                with self._autocast():
                    output = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=labels,
                    )
                    loss = output.loss / int(self.settings.gradient_accumulation_steps)
                scaler.scale(loss).backward()
                epoch_loss.append(float(output.loss.detach().cpu()))
                for head, value in output.head_losses.items():
                    head_losses[head].append(float(value.detach().cpu()))
                update = (batch_index + 1) % int(self.settings.gradient_accumulation_steps) == 0
                update = update or batch_index + 1 == len(train_loader)
                if update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.settings.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if torch.cuda.is_available():
                        peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated())
                    if self.settings.save_steps and global_step % self.settings.save_steps == 0:
                        self._save_checkpoint(
                            global_step,
                            optimizer,
                            scheduler,
                            scaler,
                            {
                                "epoch": epoch,
                                "batch_in_epoch": batch_index + 1,
                                "global_step": global_step,
                                "best_validation_score": best_score,
                                "best_validation_loss": best_loss,
                                "best_checkpoint": best_checkpoint,
                                "early_stopping_counter": patience,
                                "history": self.history,
                                "epoch_loss": epoch_loss,
                                "epoch_head_losses": dict(head_losses),
                            },
                        )
            validation_predictions = predict_dataloader(self.model, validation_loader, self.device)
            validation_metrics = metrics_from_predictions(
                validation_predictions, self.bundle.mappings, self.thresholds
            )
            validation_loss = float(np.mean(list(validation_metrics["per_head_loss"].values())))
            score = float(validation_metrics["core_macro_f1"])
            improved = score > best_score or (
                math.isclose(score, best_score) and validation_loss < best_loss
            )
            if improved:
                best_score = score
                best_loss = validation_loss
                patience = 0
            else:
                patience += 1
            history_item = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "training_loss": float(np.mean(epoch_loss)),
                "training_head_losses": {
                    head: float(np.mean(values)) for head, values in head_losses.items()
                },
                "validation_loss": validation_loss,
                "validation_core_macro_f1": score,
                "validation_head_metrics": validation_metrics,
                "selected_as_best": improved,
            }
            self.history.append(history_item)
            state = {
                "epoch": epoch + 1,
                "batch_in_epoch": 0,
                "global_step": global_step,
                "best_validation_score": best_score,
                "best_validation_loss": best_loss,
                "best_checkpoint": best_checkpoint,
                "early_stopping_counter": patience,
                "history": self.history,
            }
            if improved:
                best_checkpoint = str(
                    self.output / f"checkpoint-{global_step:08d}-epoch-{epoch + 1:03d}"
                )
                state["best_checkpoint"] = best_checkpoint
            self._save_checkpoint(
                global_step,
                optimizer,
                scheduler,
                scaler,
                state,
                suffix=f"epoch-{epoch + 1:03d}",
            )
            write_json(self.reports / "training_history.json", {"epochs": self.history})
            write_json(self.reports / "validation_metrics.json", validation_metrics)
            write_json(
                self.reports / "checkpoint_inventory.json", checkpoint_inventory(self.output)
            )
            if patience >= self.settings.early_stopping_patience:
                break
            start_batch = 0
        return {
            "best_checkpoint": best_checkpoint,
            "best_validation_score": best_score,
            "best_validation_loss": best_loss,
            "global_step": global_step,
            "epochs_completed": len(self.history),
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu,
            "resume_events": self.resume_events,
            "history": self.history,
        }

    def verify_resume_probe(self, checkpoint: str | Path) -> dict[str, Any]:
        """Reload state and execute one extra optimizer step for smoke-test validation."""

        self._load_model_weights(Path(checkpoint))
        loader, _ = self._dataloaders()
        optimizer, scheduler, scaler = self._optimizer_scheduler(len(loader))
        state = load_training_state(
            checkpoint, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        batch = next(iter(loader))
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        output = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels={head: value.to(self.device) for head, value in batch["labels"].items()},
        )
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_grad_norm)
        optimizer.step()
        scheduler.step()
        report = {
            "status": "PASS",
            "restored_global_step": state["global_step"],
            "additional_steps": 1,
            "loss": float(output.loss.detach().cpu()),
        }
        write_json(self.reports / "smoke_resume_probe.json", report)
        gc.collect()
        return report
