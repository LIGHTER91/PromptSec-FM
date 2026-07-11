from __future__ import annotations

import json
from collections.abc import Iterable

from promptsec.data.config import SourceConfig
from promptsec.data.importers.base import BaseImporter, RawRecord, uncertain_annotations
from promptsec.data.pipeline import build_dataset


class FixtureImporter(BaseImporter):
    def transform(self, raw: RawRecord) -> Iterable[dict]:
        text = str(raw.payload["prompt"])
        yield self.build_record(
            raw,
            text=text,
            source_text=text,
            annotations=uncertain_annotations(),
            original_labels={"legacy_label": raw.payload["legacy_label"]},
            mapping_status="NEEDS_REVIEW",
            field_mappings=[{"source": "prompt", "target": "content.text", "method": "COPY_EXACT"}],
            review_reasons=["Legacy benign labels do not determine instruction presence."],
        )


def _write_config(tmp_path) -> SourceConfig:
    path = tmp_path / "fixture.toml"
    path.write_text(
        """
schema_version = "0.1"

[source]
id = "fixture"
name = "Fixture"
homepage = "https://example.test/fixture"
repository = "https://example.test/fixture.git"
version = "1"
revision = "1111111111111111111111111111111111111111"
license_manifest = "manifests/sources/fixture.json"
importer = "tests.data.test_pipeline:FixtureImporter"

[scenario]
language = "en"
delivery_mode = "DIRECT"
source_role = "USER"
content_origin = "CHAT_MESSAGE"
ingestion_path = "CHAT_INPUT"
modality = "TEXT"
source_integrity = "UNKNOWN"

[fields]
text = ["prompt"]
labels = ["legacy_label"]
record_id = ["source_id"]

[[artifacts]]
id = "records"
split = "test"
format = "jsonl"
url = "https://example.test/records.jsonl"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return SourceConfig.load(path)


def test_pipeline_writes_valid_deterministic_jsonl_and_report(tmp_path) -> None:
    config = _write_config(tmp_path)
    raw_path = tmp_path / "records.jsonl"
    raw_path.write_text(
        json.dumps(
            {"source_id": "row-1", "prompt": "Please summarize this.", "legacy_label": "SAFE"}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "canonical.jsonl"
    report_path = tmp_path / "report.json"
    importer = FixtureImporter(config, imported_at="2026-07-11T12:00:00Z")

    report = build_dataset(
        importer,
        {"records": raw_path},
        output,
        report_path=report_path,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert report.records == 1
    assert report.mapping_statuses == {"NEEDS_REVIEW": 1}
    assert report.verdicts == {"UNCERTAIN": 1}
    assert report.output_sha256
    assert record["metadata"]["dataset_provenance"]["source_record"]["original_labels"] == {
        "legacy_label": "SAFE"
    }
    assert "SAFE" not in record["annotations"].values()
    assert json.loads(report_path.read_text(encoding="utf-8"))["records"] == 1


def test_canonical_id_does_not_depend_on_source_labels(tmp_path) -> None:
    config = _write_config(tmp_path)
    importer = FixtureImporter(config, imported_at="2026-07-11T12:00:00Z")
    artifact = config.artifacts[0]
    common = {"source_id": "stable", "prompt": "same text"}
    first = RawRecord(artifact, "records.jsonl", "test", 0, {**common, "legacy_label": "SAFE"})
    second = RawRecord(
        artifact, "records.jsonl", "test", 0, {**common, "legacy_label": "INJECTION"}
    )

    assert importer.canonical_id(first) == importer.canonical_id(second)
