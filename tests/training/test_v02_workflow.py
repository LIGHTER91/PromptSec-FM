from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from promptsec.training.colab_config import load_training_config
from promptsec.training.retention import (
    apply_checkpoint_retention,
    plan_checkpoint_retention,
    reset_isolated_smoke_directories,
)

ROOT = Path(__file__).resolve().parents[2]


def test_v01_configuration_remains_v01_behavior_and_fingerprint() -> None:
    path = ROOT / "configs" / "xlmr_multitask_colab_v0.1.yaml"
    settings = load_training_config(path)
    values = settings.as_dict()
    assert settings.schema_version == "0.1"
    assert "schema_version" not in values
    assert "counterfactual_loss_weight" not in values
    expected = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert settings.fingerprint() == expected
    assert settings.use_pair_aware_sampler is False
    assert settings.verdict_decoding_strategy == "DIRECT_HEAD"


def test_v02_configuration_is_explicit_and_isolated() -> None:
    settings = load_training_config(ROOT / "configs" / "xlmr_multitask_colab_v0.2.yaml")
    assert settings.schema_version == "0.2"
    assert settings.experiment_name == "relational"
    assert settings.use_pair_aware_sampler
    assert settings.require_source_commit_match
    assert settings.validation_selection_metric == "ROBUST_VALIDATION_SCORE"


def test_cli_smoke_resume_default_is_no_resume_for_v02() -> None:
    script = (ROOT / "scripts" / "train_xlmr_multitask.py").read_text(encoding="utf-8")
    assert 'args.resume = not (args.smoke_test and settings.schema_version == "0.2")' in script
    assert '"--reset-smoke-test",\n        action=argparse.BooleanOptionalAction' in script


def test_smoke_reset_is_bounded_to_v02(tmp_path) -> None:
    safe = tmp_path / "xlmr-base-multitask-v0.2-relational" / "smoke-test"
    safe.mkdir(parents=True)
    (safe / "partial.txt").write_text("partial", encoding="utf-8")
    assert reset_isolated_smoke_directories([safe]) == [str(safe.resolve())]
    assert not safe.exists()
    unsafe = tmp_path / "xlmr-base-multitask-v0.1" / "smoke-test"
    with pytest.raises(ValueError, match="refusing unsafe"):
        reset_isolated_smoke_directories([unsafe])


def test_checkpoint_retention_dry_run_is_checksum_gated(tmp_path, monkeypatch) -> None:
    root = tmp_path / "xlmr-base-multitask-v0.2-relational"
    checkpoints = []
    for step in (1, 2, 3):
        checkpoint = root / f"checkpoint-{step:08d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "payload.bin").write_bytes(bytes([step]))
        checkpoints.append(checkpoint)

    def fake_verify(path):
        step = int(Path(path).name.split("-")[1])
        return {"trainer_state": {"global_step": step}}

    monkeypatch.setattr("promptsec.training.retention.verify_checkpoint", fake_verify)
    plan = plan_checkpoint_retention(root, best_checkpoint=checkpoints[0], keep_last_n_complete=1)
    assert plan["preserve"] == sorted(
        [str(checkpoints[0].resolve()), str(checkpoints[2].resolve())]
    )
    export = root / "best_model"
    export.mkdir()
    files = {}
    for name in ("config.json", "tokenizer_config.json", "label_mappings.json"):
        path = export / name
        path.write_text("{}", encoding="utf-8")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (export / "checksums.json").write_text(json.dumps(files), encoding="utf-8")
    result = apply_checkpoint_retention(
        plan,
        verified_best_export=export,
        dry_run=True,
        verify_inference_load=False,
    )
    assert result["status"] == "DRY_RUN"
    assert checkpoints[1].exists()
    (export / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        apply_checkpoint_retention(
            plan,
            verified_best_export=export,
            dry_run=True,
            verify_inference_load=False,
        )


def test_notebook_is_valid_deterministic_and_full_training_disabled(tmp_path) -> None:
    builder = ROOT / "scripts" / "build_xlmr_colab_notebook_v0_2.py"
    notebook_path = ROOT / "notebooks" / "PromptSec_FM_XLMR_Multitask_Colab_v0_2.ipynb"
    subprocess.run([sys.executable, str(builder)], cwd=ROOT, check=True, capture_output=True)
    first = notebook_path.read_bytes()
    subprocess.run([sys.executable, str(builder)], cwd=ROOT, check=True, capture_output=True)
    second = notebook_path.read_bytes()
    assert first == second
    notebook = nbformat.read(notebook_path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "START_V0_2_TRAINING = False" in source
    assert "RUN_FOCAL_EXPERIMENT = False" in source
    assert "BILINGUAL_FINAL_SILVER_MODEL = False" in source
    assert "--reset-smoke-test" in source and "--no-resume" in source
    assert len([cell for cell in notebook.cells if cell.cell_type == "markdown"]) >= 21


def test_canonical_release_hashes_are_unchanged_when_available() -> None:
    root = ROOT / "data" / "generated" / "policybench-codex-v0.1"
    if not root.is_dir():
        pytest.skip("immutable release is not locally available")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.jsonl")
    }
    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.jsonl")
    }
    assert before == after
    assert manifest["data_quality"] == "SILVER_VALIDATED"
    assert manifest["human_validation_status"] == "PENDING"
    assert manifest["automatic_gold_records"] == 0
