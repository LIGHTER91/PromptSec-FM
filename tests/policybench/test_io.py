from __future__ import annotations

from pathlib import Path

import pytest

from promptsec.policybench.io import (
    PolicyBenchIOError,
    iter_jsonl,
    loads_json_object,
    safe_child,
    write_jsonl,
    write_named_checksums,
)


def test_strict_json_rejects_non_utf8_constants_and_oversize() -> None:
    with pytest.raises(PolicyBenchIOError, match="strict UTF-8"):
        loads_json_object(b'\xff{"ok":true}')
    with pytest.raises(PolicyBenchIOError, match="non-standard JSON constant"):
        loads_json_object('{"value":NaN}')
    with pytest.raises(PolicyBenchIOError, match="duplicate JSON object key"):
        loads_json_object('{"value":1,"value":2}')
    with pytest.raises(PolicyBenchIOError, match="maximum size"):
        loads_json_object('{"value":1}', maximum_bytes=2)


def test_safe_child_rejects_path_traversal(tmp_path: Path) -> None:
    assert safe_child(tmp_path, "scenario/attempt.json").is_relative_to(tmp_path)
    with pytest.raises(PolicyBenchIOError, match="unsafe artifact path"):
        safe_child(tmp_path, "../secret.txt")
    with pytest.raises(PolicyBenchIOError, match="unsafe artifact path"):
        safe_child(tmp_path, tmp_path / "absolute.json")


def test_jsonl_and_checksums_are_deterministic_and_duplicate_ids_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    records = [{"id": "b", "text": "é"}, {"id": "a", "text": "🙂"}]
    write_jsonl(path, records)
    first = path.read_bytes()
    write_jsonl(path, records)
    assert path.read_bytes() == first
    assert iter_jsonl(path) == records

    checksum_path = write_named_checksums(tmp_path, ["records.jsonl"])
    assert checksum_path.read_text(encoding="utf-8").endswith("  records.jsonl\n")

    write_jsonl(path, [{"id": "same"}, {"id": "same"}])
    with pytest.raises(PolicyBenchIOError, match="duplicate record id"):
        iter_jsonl(path)
