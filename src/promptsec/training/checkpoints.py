"""Atomic, checksummed, fingerprint-compatible Drive checkpoints."""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import __version__ as transformers_version

from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import read_json_object, write_json

REQUIRED_CHECKPOINT_FILES = (
    "model/config.json",
    "tokenizer/tokenizer_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.pt",
    "label_mappings.json",
    "training_config.json",
)


class IncompleteCheckpointError(ValueError):
    pass


class IncompatibleCheckpointError(ValueError):
    pass


def capture_rng_states() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng_states(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all(value["cuda"])


def _all_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "checkpoint_manifest.json"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _json_manifest_scalar(value: Any) -> Any:
    """Keep manifests strict JSON while binary trainer state retains sentinels."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_checkpoint_atomic(
    destination: str | Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    trainer_state: Mapping[str, Any],
    label_mappings: Mapping[str, Any],
    training_config: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    environment: Mapping[str, Any],
    run_kind: str,
) -> Path:
    """Write all state to a sibling temporary directory and rename only when complete."""

    final = Path(destination).resolve()
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary / "model", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(scaler.state_dict() if scaler is not None else None, temporary / "scaler.pt")
        torch.save(
            {**trainer_state, "rng_states": capture_rng_states()},
            temporary / "trainer_state.pt",
        )
        write_json(temporary / "label_mappings.json", label_mappings)
        write_json(temporary / "training_config.json", training_config)
        missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (temporary / name).is_file()]
        if missing:
            raise IncompleteCheckpointError(f"checkpoint write is missing files: {missing}")
        checksums = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in _all_files(temporary)
        }
        write_json(
            temporary / "checkpoint_manifest.json",
            {
                "schema_version": "0.1",
                "status": "COMPLETE",
                "run_kind": run_kind,
                "compatibility": dict(compatibility),
                "trainer_state": {
                    key: _json_manifest_scalar(value)
                    for key, value in trainer_state.items()
                    if isinstance(value, (str, int, float, bool, type(None)))
                },
                "environment": dict(environment),
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
                "files": checksums,
            },
        )
        verify_checkpoint(temporary)
        os.rename(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def verify_checkpoint(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise IncompleteCheckpointError(f"checkpoint manifest missing: {root}")
    manifest = read_json_object(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise IncompleteCheckpointError(f"checkpoint is not complete: {root}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise IncompleteCheckpointError("checkpoint files mapping is absent")
    for name, digest in files.items():
        candidate = root / name
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise IncompleteCheckpointError(f"checkpoint checksum mismatch: {name}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if name not in files]
    if missing:
        raise IncompleteCheckpointError(f"checkpoint manifest omits required files: {missing}")
    return manifest


def find_latest_compatible_checkpoint(
    root: str | Path,
    *,
    compatibility: Mapping[str, Any],
    run_kind: str,
) -> Path | None:
    directory = Path(root)
    if not directory.is_dir():
        return None
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for path in directory.glob("checkpoint-*"):
        try:
            match = re.match(r"^checkpoint-(\d+)", path.name)
            if match is None:
                continue
            step = int(match.group(1))
            manifest = verify_checkpoint(path)
        except (ValueError, IncompleteCheckpointError):
            continue
        complete.append((step, path, manifest))
    if not complete:
        return None
    compatible = [
        item
        for item in complete
        if item[2].get("compatibility") == dict(compatibility)
        and item[2].get("run_kind") == run_kind
    ]
    if not compatible:
        raise IncompatibleCheckpointError(
            "complete checkpoints exist, but none match dataset/configuration/run-kind fingerprints"
        )
    return max(compatible, key=lambda item: (item[0], item[1].name))[1]


def load_training_state(
    checkpoint: str | Path,
    *,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
) -> dict[str, Any]:
    root = Path(checkpoint)
    verify_checkpoint(root)
    optimizer.load_state_dict(
        torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    scaler_state = torch.load(root / "scaler.pt", map_location="cpu", weights_only=True)
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    state = torch.load(root / "trainer_state.pt", map_location="cpu", weights_only=False)
    restore_rng_states(state.pop("rng_states"))
    return state


def checkpoint_inventory(root: str | Path) -> dict[str, Any]:
    items = []
    for path in sorted(Path(root).glob("checkpoint-*")) if Path(root).is_dir() else []:
        try:
            manifest = verify_checkpoint(path)
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            items.append(
                {
                    "path": str(path),
                    "status": "COMPLETE",
                    "size_bytes": size,
                    "manifest": manifest,
                }
            )
        except IncompleteCheckpointError as error:
            items.append({"path": str(path), "status": "INCOMPLETE", "error": str(error)})
    return {
        "checkpoints": items,
        "total_complete_size_bytes": sum(
            item.get("size_bytes", 0) for item in items if item["status"] == "COMPLETE"
        ),
    }
