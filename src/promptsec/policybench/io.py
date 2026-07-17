"""Deterministic, strict file I/O for PolicyBench artifacts.

Generated model output is untrusted.  This module never evaluates content, rejects
non-standard JSON constants, and confines computed child paths to an explicit root.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from promptsec.data.hashing import canonical_json_bytes, sha256_file


class PolicyBenchIOError(ValueError):
    """Raised when a PolicyBench artifact cannot be read or written safely."""


def _reject_constant(value: str) -> None:
    raise PolicyBenchIOError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyBenchIOError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def loads_json_object(data: str | bytes, *, maximum_bytes: int | None = None) -> dict[str, Any]:
    """Decode one strict UTF-8 JSON object with an optional encoded-size limit."""

    encoded = data.encode("utf-8") if isinstance(data, str) else data
    if maximum_bytes is not None and len(encoded) > maximum_bytes:
        raise PolicyBenchIOError(
            f"JSON payload exceeds maximum size ({len(encoded)} > {maximum_bytes} bytes)"
        )
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PolicyBenchIOError(f"JSON payload is not strict UTF-8: {error}") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except json.JSONDecodeError as error:
        raise PolicyBenchIOError(f"malformed JSON: {error}") from error
    if not isinstance(value, dict):
        raise PolicyBenchIOError("JSON payload must be an object")
    return value


def read_json_object(path: str | Path, *, maximum_bytes: int | None = None) -> dict[str, Any]:
    artifact = Path(path)
    try:
        data = artifact.read_bytes()
    except OSError as error:
        raise PolicyBenchIOError(f"cannot read JSON object {artifact}: {error}") from error
    return loads_json_object(data, maximum_bytes=maximum_bytes)


def iter_jsonl(path: str | Path, *, maximum_line_bytes: int = 2_000_000) -> list[dict[str, Any]]:
    """Read a JSONL file strictly and reject blank lines and duplicate record IDs."""

    artifact = Path(path)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with artifact.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if len(raw_line) > maximum_line_bytes:
                    raise PolicyBenchIOError(
                        f"{artifact}:{line_number}: line exceeds {maximum_line_bytes} bytes"
                    )
                if not raw_line.strip():
                    raise PolicyBenchIOError(f"{artifact}:{line_number}: blank lines are forbidden")
                record = loads_json_object(raw_line, maximum_bytes=maximum_line_bytes)
                record_id = record.get("id")
                if isinstance(record_id, str):
                    if record_id in seen_ids:
                        raise PolicyBenchIOError(
                            f"{artifact}:{line_number}: duplicate record id {record_id!r}"
                        )
                    seen_ids.add(record_id)
                records.append(record)
    except OSError as error:
        raise PolicyBenchIOError(f"cannot read JSONL {artifact}: {error}") from error
    return records


def safe_child(root: str | Path, relative: str | Path) -> Path:
    """Resolve a relative artifact path and prove that it remains below ``root``."""

    base = Path(root).resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise PolicyBenchIOError(f"unsafe artifact path: {relative}")
    candidate = (base / candidate_relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise PolicyBenchIOError(f"artifact path escapes root: {relative}")
    return candidate


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise PolicyBenchIOError(f"cannot atomically write {path}: {error}") from error


def write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    if pretty:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    else:
        data = canonical_json_bytes(value) + b"\n"
    _atomic_write(Path(path), data)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    ordered = list(records)
    data = b"".join(canonical_json_bytes(record) + b"\n" for record in ordered)
    _atomic_write(Path(path), data)


def write_named_checksums(root: str | Path, names: Iterable[str]) -> Path:
    base = Path(root).resolve()
    lines: list[str] = []
    for name in sorted(set(names)):
        path = safe_child(base, name)
        if not path.is_file():
            raise PolicyBenchIOError(f"cannot checksum missing artifact: {name}")
        lines.append(f"{sha256_file(path)}  {Path(name).as_posix()}")
    destination = safe_child(base, "checksums.sha256")
    _atomic_write(destination, ("\n".join(lines) + "\n").encode("utf-8"))
    return destination


__all__ = [
    "PolicyBenchIOError",
    "iter_jsonl",
    "loads_json_object",
    "read_json_object",
    "safe_child",
    "write_json",
    "write_jsonl",
    "write_named_checksums",
]
