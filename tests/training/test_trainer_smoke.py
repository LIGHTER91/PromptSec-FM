from __future__ import annotations

from pathlib import Path

from transformers import XLMRobertaConfig, XLMRobertaModel

from promptsec.training.colab_config import TrainingSettings
from promptsec.training.dataset import TrainingDatasetBundle
from promptsec.training.multitask_model import (
    PromptSecMultitaskConfig,
    PromptSecMultitaskModel,
)
from promptsec.training.serialization import SPECIAL_TOKENS
from promptsec.training.trainer import MultitaskTrainer


def test_tiny_cpu_training_checkpoint_reload_and_resume_probe(
    tmp_path, fake_tokenizer, label_mappings, record_factory
) -> None:
    encoder_config = XLMRobertaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=96,
    )
    model_config = PromptSecMultitaskConfig(
        encoder_config=encoder_config.to_dict(),
        head_dimensions={head: len(mapping.labels) for head, mapping in label_mappings.items()},
        label_mappings={head: mapping.to_dict() for head, mapping in label_mappings.items()},
        special_tokens=list(SPECIAL_TOKENS),
    )
    model = PromptSecMultitaskModel(model_config, encoder=XLMRobertaModel(encoder_config))
    records = [record_factory(index) for index in range(6)]
    bundle = TrainingDatasetBundle(
        root=tmp_path / "fixture-release",
        manifest={"records": 6000},
        manifest_sha256="a" * 64,
        split_hashes={"train": "b" * 64, "validation": "c" * 64},
        records_by_split={"train": records[:4], "validation": records[4:]},
        mappings=label_mappings,
        integrity_report={"validation_status": "PASS"},
        split_audit={"leakage_detected": False},
        annotation_schema=Path("schemas/promptsec-annotation-v1.schema.json"),
    )
    settings = TrainingSettings(
        model_name="tiny-local-xlm-roberta",
        max_length=64,
        epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        save_steps=0,
        seed=19,
    )
    trainer = MultitaskTrainer(
        model=model,
        tokenizer=fake_tokenizer,
        bundle=bundle,
        settings=settings,
        output=tmp_path / "checkpoints" / "smoke-test",
        reports=tmp_path / "reports" / "smoke-test",
        environment={"precision": "fp32"},
        resume=True,
        smoke_test=True,
    )
    result = trainer.train()
    checkpoint = Path(result["best_checkpoint"])
    assert checkpoint.is_dir()
    assert (checkpoint / "checkpoint_manifest.json").is_file()
    probe = trainer.verify_resume_probe(checkpoint)
    assert probe["status"] == "PASS"
    assert probe["additional_steps"] == 1
    assert (tmp_path / "reports" / "smoke-test" / "training_history.json").is_file()
    assert not (tmp_path / "checkpoints" / "full").exists()
