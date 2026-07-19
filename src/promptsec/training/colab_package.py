"""Deterministic, minimal, integrity-checked PolicyBench Colab archives."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from promptsec.baselines.dataset import release_file_hashes
from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import write_json
from promptsec.policybench.splitting import PUBLISHED_SPLITS
from promptsec.policybench.validation import (
    iter_dataset_records,
    validate_record_collection,
    validate_release_directory,
)

RELEASE_FILES = (
    "checksums.sha256",
    "manifest.json",
    "quality_report.json",
    "split_report.json",
    *(f"{split}.jsonl" for split in PUBLISHED_SPLITS),
)
SCHEMA_FILES = (
    "promptsec-annotation-v1.schema.json",
    "promptsec-dataset-record-v0.1.schema.json",
    "promptsec-policybench-record-v0.1.schema.json",
)


def _manifest_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def package_policybench_for_colab(
    dataset: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(dataset).resolve()
    destination = Path(output).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Colab package output must be outside the immutable source release")
    sha_path = destination.with_name(f"{destination.name}.sha256")
    external_manifest_path = destination.parent / "colab_input_manifest.json"
    before = release_file_hashes(source)
    records = iter_dataset_records(source)
    validation = validate_release_directory(source, records)
    if validation["validation_status"] != "PASS":
        raise ValueError(f"source release validation failed: {validation['errors']}")
    if len(records) != 6000:
        raise ValueError("source release must contain exactly 6,000 records")
    canonical_validation = validate_record_collection(records)
    if canonical_validation["validation_status"] != "PASS":
        raise ValueError(
            "source canonical validation failed for "
            f"{canonical_validation['invalid_records']} records"
        )
    repository = Path(__file__).resolve().parents[3]
    release_id = source.name
    entries: dict[str, bytes] = {}
    file_hashes: dict[str, str] = {}
    for name in RELEASE_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"required Colab release input is missing: {name}")
        relative = name
        entries[f"{release_id}/{relative}"] = path.read_bytes()
        file_hashes[relative] = sha256_file(path)
    for name in SCHEMA_FILES:
        path = repository / "schemas" / name
        if not path.is_file():
            raise FileNotFoundError(f"required Colab schema is missing: {name}")
        relative = f"schemas/{name}"
        entries[f"{release_id}/{relative}"] = path.read_bytes()
        file_hashes[relative] = sha256_file(path)
    package_manifest = {
        "schema_version": "0.1",
        "package_type": "PromptSec-PolicyBench Colab training input",
        "release_id": release_id,
        "records": 6000,
        "dataset_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_release_validation": validation,
        "source_canonical_validation": canonical_validation,
        "files": dict(sorted(file_hashes.items())),
        "excluded": [
            "raw attempts",
            "quarantine",
            "credentials",
            "caches",
            "review annotations",
            "Git metadata",
            "model artifacts",
            "operational logs",
        ],
    }
    entries[f"{release_id}/colab_input_manifest.json"] = _manifest_bytes(package_manifest)
    if destination.exists() and not overwrite:
        if sha_path.is_file() and external_manifest_path.is_file():
            declared = sha_path.read_text(encoding="utf-8").split()[0]
            existing = sha256_file(destination)
            external = json.loads(external_manifest_path.read_text(encoding="utf-8"))
            if (
                declared == existing
                and external.get("dataset_manifest_sha256")
                == package_manifest["dataset_manifest_sha256"]
            ):
                if before != release_file_hashes(source):
                    raise RuntimeError("source release changed during Colab packaging")
                return {**external, "status": "REUSED_COMPATIBLE"}
        raise FileExistsError("Colab archive exists but is not safely reusable; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(entries):
                archive.writestr(
                    _zip_info(name),
                    entries[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        archive_sha256 = sha256_file(temporary)
        result = {
            **package_manifest,
            "archive": str(destination),
            "archive_sha256": archive_sha256,
            "archive_bytes": temporary.stat().st_size,
            "status": "CREATED",
        }
        after = release_file_hashes(source)
        if before != after:
            raise RuntimeError("source release changed during Colab packaging")
        os.replace(temporary, destination)
        temporary = None
        _atomic_text(sha_path, f"{archive_sha256}  {destination.name}\n")
        write_json(external_manifest_path, result)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
