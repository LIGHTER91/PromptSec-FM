"""Dependency-free readers for the raw formats used by initial sources."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from promptsec.data.config import ArtifactConfig


class ReaderError(ValueError):
    """Raised when a raw artifact cannot be decoded into object records."""


def _records_at_path(value: Any, path: str | None) -> Any:
    current = value
    if path:
        for component in path.split("."):
            if not isinstance(current, dict) or component not in current:
                raise ReaderError(f"JSON records_path {path!r} does not exist")
            current = current[component]
    return current


def _ensure_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReaderError(f"{location}: expected a JSON object, got {type(value).__name__}")
    return value


def read_records(path: str | Path, artifact: ArtifactConfig) -> Iterator[dict[str, Any]]:
    raw_path = Path(path)
    try:
        if artifact.format in {"csv", "tsv"}:
            delimiter = "," if artifact.format == "csv" else "\t"
            with raw_path.open("r", encoding=artifact.encoding, newline="") as stream:
                yield from csv.DictReader(stream, delimiter=delimiter)
            return

        if artifact.format == "jsonl":
            with raw_path.open("r", encoding=artifact.encoding) as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        yield _ensure_object(json.loads(line), f"{raw_path}:{line_number}")
            return

        if artifact.format == "json":
            value = json.loads(raw_path.read_text(encoding=artifact.encoding))
            value = _records_at_path(value, artifact.records_path)
            if isinstance(value, dict):
                for record_id, record in value.items():
                    item = _ensure_object(record, f"{raw_path}:{record_id}").copy()
                    item.setdefault("_source_key", record_id)
                    yield item
                return
            if not isinstance(value, list):
                raise ReaderError(f"{raw_path}: expected a JSON array or object of records")
            for index, record in enumerate(value):
                yield _ensure_object(record, f"{raw_path}[{index}]")
            return

        if artifact.format == "text":
            with raw_path.open("r", encoding=artifact.encoding) as stream:
                for line_number, line in enumerate(stream, 1):
                    yield {"text": line.rstrip("\r\n"), "_source_line": line_number}
            return
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise ReaderError(f"cannot read {raw_path}: {exc}") from exc

    raise ReaderError(f"unsupported raw format: {artifact.format}")
