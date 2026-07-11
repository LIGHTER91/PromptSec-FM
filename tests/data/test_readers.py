from __future__ import annotations

import json

from promptsec.data.config import ArtifactConfig
from promptsec.data.readers import read_records


def _artifact(format_name: str, *, records_path: str | None = None) -> ArtifactConfig:
    return ArtifactConfig(
        id="fixture",
        split="test",
        format=format_name,
        url="https://example.test/fixture",
        records_path=records_path,
    )


def test_jsonl_reader_preserves_objects_and_unicode(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [{"id": 1, "text": "café 🙂"}, {"id": 2, "text": "unchanged"}]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    assert list(read_records(path, _artifact("jsonl"))) == rows


def test_json_reader_supports_nested_record_arrays(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"payload": {"rows": [{"text": "one"}]}}), encoding="utf-8")

    assert list(read_records(path, _artifact("json", records_path="payload.rows"))) == [
        {"text": "one"}
    ]


def test_csv_reader_uses_header_names(tmp_path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("prompt,label\nhello,benign\n", encoding="utf-8")

    assert list(read_records(path, _artifact("csv"))) == [{"prompt": "hello", "label": "benign"}]
