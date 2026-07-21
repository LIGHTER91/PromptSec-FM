"""Checksum-gated v0.2 checkpoint retention with dry-run support."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import read_json_object, write_json
from promptsec.training.checkpoints import verify_checkpoint


def verify_export(export: str | Path, *, verify_inference_load: bool = True) -> dict[str, Any]:
    root = Path(export)
    checksums_path = root / "checksums.json"
    if not checksums_path.is_file():
        raise ValueError("best_model checksums.json is missing")
    checksums = read_json_object(checksums_path)
    required = ("config.json", "tokenizer_config.json", "label_mappings.json")
    for name in required:
        if name not in checksums:
            raise ValueError(f"best_model export omits required file: {name}")
    for name, digest in checksums.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"best_model checksum mismatch: {name}")
    inference_status = "NOT_REQUESTED"
    if verify_inference_load:
        import gc

        from transformers import AutoTokenizer

        from promptsec.training.multitask_model import PromptSecMultitaskModel

        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = PromptSecMultitaskModel.from_pretrained(root, local_files_only=True)
        if not tokenizer or not model.config.label_mappings:
            raise ValueError("best_model inference load did not restore tokenizer/mappings")
        exported_mappings = read_json_object(root / "label_mappings.json").get("heads", {})
        if exported_mappings != model.config.label_mappings:
            raise ValueError("best_model label mappings differ from the model configuration")
        vocabulary = tokenizer.get_vocab()
        missing_tokens = [token for token in model.config.special_tokens if token not in vocabulary]
        if missing_tokens:
            raise ValueError(f"best_model tokenizer is missing special tokens: {missing_tokens}")
        del model, tokenizer
        gc.collect()
        inference_status = "PASS"
    return {
        "status": "PASS",
        "files_verified": len(checksums),
        "inference_load": inference_status,
    }


def plan_checkpoint_retention(
    checkpoint_root: str | Path,
    *,
    best_checkpoint: str | Path,
    keep_last_n_complete: int = 2,
    compatibility: Mapping[str, Any] | None = None,
    run_kind: str | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    best = Path(best_checkpoint).resolve()
    if "v0.1" in root.as_posix().lower():
        raise ValueError("v0.1 checkpoint roots are permanently outside v0.2 retention scope")
    if best.parent != root:
        raise ValueError("best checkpoint must be an immediate child of the v0.2 checkpoint root")
    best_manifest = verify_checkpoint(best)
    if compatibility is not None and (
        best_manifest.get("compatibility") != dict(compatibility)
        or (run_kind is not None and best_manifest.get("run_kind") != run_kind)
    ):
        raise ValueError("best checkpoint is not compatible with the current v0.2 run")
    complete: list[tuple[int, Path]] = []
    invalid: list[dict[str, str]] = []
    for path in sorted(root.glob("checkpoint-*")) if root.is_dir() else []:
        try:
            manifest = verify_checkpoint(path)
            if compatibility is not None and (
                manifest.get("compatibility") != dict(compatibility)
                or (run_kind is not None and manifest.get("run_kind") != run_kind)
            ):
                invalid.append(
                    {"path": str(path), "reason": "incompatible checkpoint left untouched"}
                )
                continue
            step = int(manifest.get("trainer_state", {}).get("global_step") or 0)
            complete.append((step, path.resolve()))
        except (ValueError, OSError) as error:
            invalid.append({"path": str(path), "reason": str(error)})
    latest = [path for _, path in sorted(complete, reverse=True)[:keep_last_n_complete]]
    preserve = {best, *latest}
    remove = [path for _, path in complete if path not in preserve]
    return {
        "checkpoint_root": str(root),
        "best_checkpoint": str(best),
        "preserve": sorted(map(str, preserve)),
        "remove": sorted(map(str, remove)),
        "invalid_untouched": invalid,
        "checksums_validated_before_planning": True,
    }


def apply_checkpoint_retention(
    plan: dict[str, Any],
    *,
    verified_best_export: str | Path,
    dry_run: bool = True,
    manifest_path: str | Path | None = None,
    verify_inference_load: bool = True,
) -> dict[str, Any]:
    verification = verify_export(verified_best_export, verify_inference_load=verify_inference_load)
    root = Path(plan["checkpoint_root"]).resolve()
    removed: list[str] = []
    reclaimed = 0
    for value in plan.get("remove", []):
        path = Path(value).resolve()
        if path.parent != root or "v0.1" in path.as_posix().lower():
            raise ValueError(f"unsafe checkpoint retention target: {path}")
        verify_checkpoint(path)
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        reclaimed += size
        if not dry_run:
            shutil.rmtree(path)
        removed.append(str(path))
    result = {
        **plan,
        "dry_run": dry_run,
        "verified_best_export": verification,
        "removed_or_planned": removed,
        "reclaimed_bytes": reclaimed,
        "status": "DRY_RUN" if dry_run else "COMPLETE",
    }
    if manifest_path is not None:
        write_json(manifest_path, result)
    return result


def reset_isolated_smoke_directories(paths: Sequence[str | Path]) -> list[str]:
    """Delete only explicit v0.2 smoke-test roots."""

    removed = []
    for value in paths:
        path = Path(value).resolve()
        normalized = path.as_posix().lower()
        if path.name != "smoke-test" or "v0.2" not in normalized or "v0.1" in normalized:
            raise ValueError(f"refusing unsafe smoke reset outside an isolated v0.2 root: {path}")
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed
