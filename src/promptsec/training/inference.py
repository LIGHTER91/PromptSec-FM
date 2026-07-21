"""Standalone inference entry point for an exported PromptSec multi-task model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from promptsec.training.labels import LabelMapping
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
        decoding_path = root / "verdict_decoding_configuration.json"
        self.verdict_decoding = (
            json.loads(decoding_path.read_text(encoding="utf-8"))
            if decoding_path.is_file()
            else {"selected_strategy": "DIRECT_HEAD", "selected_alpha": 1.0}
        )
        training_path = root / "training_configuration.json"
        self.training_configuration = (
            json.loads(training_path.read_text(encoding="utf-8")) if training_path.is_file() else {}
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    def predict(self, context_record: dict[str, Any]) -> dict[str, Any]:
        encoded = encode_with_section_budget(
            self.tokenizer,
            record_sections(context_record),
            max_length=int(
                getattr(
                    self.model.config,
                    "promptsec_max_length",
                    getattr(self.model.config, "max_length", 512),
                )
            ),
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=torch.tensor([encoded.input_ids], device=self.device),
                attention_mask=torch.tensor([encoded.attention_mask], device=self.device),
            )
        decoded = self.model.decode_predictions(output.logits, threshold=self.thresholds)
        from promptsec.training.verdict import decode_verdict_probabilities

        mappings = {
            head: LabelMapping(
                head=head,
                labels=tuple(value["labels"]),
                multilabel=bool(value["multilabel"]),
            )
            for head, value in self.model.config.label_mappings.items()
        }
        verdict = decode_verdict_probabilities(
            output.logits,
            mappings,
            strategy=self.verdict_decoding.get("selected_strategy", "DIRECT_HEAD"),
            alpha=float(self.verdict_decoding.get("selected_alpha", 1.0)),
        )
        verdict_labels = mappings["prompt_injection_verdict"].labels
        verdict_output = {
            name: {
                "labels": [verdict_labels[int(values.argmax(dim=-1)[0])]],
                "probabilities": values.detach().cpu().tolist(),
            }
            for name, values in verdict.items()
        }
        return {
            "predictions": decoded,
            "direct_verdict": verdict_output["direct"],
            "derived_verdict": verdict_output["derived"],
            "final_verdict": verdict_output["final"],
            "verdict_disagreement": (
                verdict_output["direct"]["labels"] != verdict_output["derived"]["labels"]
            ),
            "decoding_strategy": self.verdict_decoding.get("selected_strategy", "DIRECT_HEAD"),
            "model_version": {
                "schema_version": self.training_configuration.get("schema_version", "0.1"),
                "experiment_name": self.training_configuration.get("experiment_name", "v0.1"),
                "source": self.model.config.name_or_path,
            },
            "preprocessing_version": self.model.config.preprocessing_version,
        }
