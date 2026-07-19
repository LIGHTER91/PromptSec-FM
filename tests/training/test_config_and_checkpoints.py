from __future__ import annotations

import json
import math

import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel, get_linear_schedule_with_warmup

from promptsec.training.checkpoints import (
    IncompatibleCheckpointError,
    IncompleteCheckpointError,
    find_latest_compatible_checkpoint,
    load_training_state,
    save_checkpoint_atomic,
    verify_checkpoint,
)
from promptsec.training.colab_config import (
    FINAL_SILVER_MODEL,
    TrainingSettings,
    require_cuda_for_full_training,
    resolve_batch_strategy,
)
from promptsec.training.multitask_model import (
    PromptSecMultitaskConfig,
    PromptSecMultitaskModel,
)


def _model(label_mappings):
    encoder_config = XLMRobertaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    config = PromptSecMultitaskConfig(
        encoder_config=encoder_config.to_dict(),
        head_dimensions={head: len(mapping.labels) for head, mapping in label_mappings.items()},
        label_mappings={head: mapping.to_dict() for head, mapping in label_mappings.items()},
        special_tokens=["<candidate>", "</candidate>"],
    )
    return PromptSecMultitaskModel(config, encoder=XLMRobertaModel(encoder_config))


def test_gpu_batch_strategy_and_cpu_gate() -> None:
    assert resolve_batch_strategy(16) == {
        "per_device_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 16,
    }
    assert resolve_batch_strategy(11)["per_device_batch_size"] == 4
    assert resolve_batch_strategy(8)["per_device_batch_size"] == 2
    require_cuda_for_full_training({"cuda_available": False}, smoke_test=True)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        require_cuda_for_full_training({"cuda_available": False}, smoke_test=False)
    with pytest.raises(ValueError, match="explicit final_silver_splits"):
        TrainingSettings(training_mode=FINAL_SILVER_MODEL).validate()


def test_atomic_checkpoint_verification_resume_and_isolation(
    tmp_path, fake_tokenizer, label_mappings
) -> None:
    model = _model(label_mappings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 2)
    compatibility = {
        "dataset_manifest_sha256": "a" * 64,
        "training_config_hash": "b" * 64,
    }
    checkpoint = save_checkpoint_atomic(
        tmp_path / "smoke" / "checkpoint-00000001",
        model=model,
        tokenizer=fake_tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        trainer_state={
            "epoch": 1,
            "global_step": 1,
            "best_validation_score": -math.inf,
            "best_validation_loss": math.inf,
        },
        label_mappings={
            "heads": {head: mapping.to_dict() for head, mapping in label_mappings.items()}
        },
        training_config={"seed": 7},
        compatibility=compatibility,
        environment={"precision": "fp32"},
        run_kind="SMOKE_TEST",
    )
    manifest = verify_checkpoint(checkpoint)
    assert manifest["status"] == "COMPLETE"
    assert manifest["trainer_state"]["best_validation_score"] is None
    assert manifest["trainer_state"]["best_validation_loss"] is None
    assert (
        find_latest_compatible_checkpoint(
            tmp_path / "smoke", compatibility=compatibility, run_kind="SMOKE_TEST"
        )
        == checkpoint
    )
    with pytest.raises(IncompatibleCheckpointError):
        find_latest_compatible_checkpoint(
            tmp_path / "smoke",
            compatibility={**compatibility, "training_config_hash": "c" * 64},
            run_kind="SMOKE_TEST",
        )
    state = load_training_state(checkpoint, optimizer=optimizer, scheduler=scheduler, scaler=None)
    assert state["global_step"] == 1
    assert state["best_validation_score"] == -math.inf
    assert state["best_validation_loss"] == math.inf
    assert not (tmp_path / "full").exists()


def test_incomplete_or_tampered_checkpoint_is_rejected(tmp_path) -> None:
    incomplete = tmp_path / "checkpoint-00000001"
    incomplete.mkdir()
    with pytest.raises(IncompleteCheckpointError):
        verify_checkpoint(incomplete)
    (incomplete / "checkpoint_manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "files": {"missing": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(IncompleteCheckpointError, match="checksum mismatch"):
        verify_checkpoint(incomplete)
