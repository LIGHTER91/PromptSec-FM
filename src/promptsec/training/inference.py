"""Standalone inference entry point for an exported PromptSec multi-task model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from promptsec.training.multitask_model import PromptSecMultitaskModel
from promptsec.training.serialization import record_sections
from promptsec.training.token_budget import encode_with_section_budget


class PromptSecPredictor:
    def __init__(self, model_directory: str | Path, *, device: str | None = None) -> None:
        root = Path(model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(root)
        self.model = PromptSecMultitaskModel.from_pretrained(root)
        threshold_path = root / "classification_thresholds.json"
        self.thresholds = (
            json.loads(threshold_path.read_text(encoding="utf-8"))
            if threshold_path.is_file()
            else {"attack_families": 0.5, "attack_objectives": 0.5}
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    def predict(self, context_record: dict[str, Any]) -> dict[str, Any]:
        encoded = encode_with_section_budget(
            self.tokenizer,
            record_sections(context_record),
            max_length=int(getattr(self.model.config, "max_length", 512)),
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=torch.tensor([encoded.input_ids], device=self.device),
                attention_mask=torch.tensor([encoded.attention_mask], device=self.device),
            )
        decoded = self.model.decode_predictions(output.logits, threshold=self.thresholds)
        return {
            "predictions": decoded,
            "derived_verdict": decoded["prompt_injection_verdict"],
            "model_version": self.model.config.name_or_path,
            "preprocessing_version": self.model.config.preprocessing_version,
        }
