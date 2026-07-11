from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptsec.data.config import SourceConfig
from promptsec.data.importers.base import ImporterError
from promptsec.data.importers.bipia import BIPIAImporter
from promptsec.data.validation import validate_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "data" / "fixtures" / "bipia"


def _inputs() -> dict[str, Path]:
    return {
        artifact_id: FIXTURES / f"{artifact_id}.json"
        for artifact_id in (
            "text_attack_train",
            "text_attack_test",
            "code_attack_train",
            "code_attack_test",
        )
    }


def _importer() -> BIPIAImporter:
    config = SourceConfig.load(REPOSITORY_ROOT / "configs" / "sources" / "bipia.toml")
    return BIPIAImporter(config, imported_at="2026-07-11T12:00:00Z")


def test_flattens_five_payload_categories_with_stable_source_ids() -> None:
    records = list(_importer().records(_inputs()))

    assert len(records) == 20
    source_records = [
        record["metadata"]["dataset_provenance"]["source_record"] for record in records
    ]
    assert source_records[0]["id"] == "text_attack_train:Information Retrieval:0"
    assert source_records[4]["id"] == "text_attack_train:Information Retrieval:4"
    assert source_records[5]["id"] == "text_attack_test:Malware Distribution:0"
    assert source_records[10]["id"] == "code_attack_train:Cookie Theft:0"
    assert len({record["id"] for record in records}) == 20
    assert source_records[0]["original_fields"] == {
        "source_id": "text_attack_train:Information Retrieval:0",
        "attack_domain": "Information Retrieval",
        "payload_type": "text",
        "instruction": "What is the capital of Brazil?",
    }


def test_maps_injected_task_role_without_inferring_objectives_from_domain_names() -> None:
    records = list(_importer().records(_inputs()))
    replacement_task = records[0]
    malware = records[5]
    code = records[10]

    assert replacement_task["annotations"]["attack_objectives"] == ["TASK_HIJACKING"]
    assert malware["annotations"]["attack_objectives"] == ["TASK_HIJACKING"]
    assert code["annotations"]["attack_objectives"] == ["TASK_HIJACKING"]
    assert replacement_task["content"]["content_origin"] == "UNKNOWN"
    assert replacement_task["content"]["modality"] == "TEXT"
    assert code["content"]["content_origin"] == "CODE_REPOSITORY"
    assert code["content"]["modality"] == "CODE"

    for record in records:
        provenance = record["metadata"]["dataset_provenance"]
        original_fields = provenance["source_record"]["original_fields"]
        assert record["content"]["delivery_mode"] == "INDIRECT"
        assert record["content"]["source_role"] == "EXTERNAL_CONTENT"
        assert record["annotations"]["attack_families"] == ["PROMPT_INJECTION"]
        assert record["annotations"]["authority_status"] == "OUTSIDE_AUTHORITY"
        assert record["annotations"]["spans"] == [
            {
                "start": 0,
                "end": len(record["content"]["text"]),
                "type": "INJECTION_PAYLOAD",
            }
        ]
        assert provenance["source_record"]["original_labels"] == {
            "attack_domain": original_fields["attack_domain"],
            "payload_type": original_fields["payload_type"],
        }
        mapping = provenance["mapping"]
        assert mapping["status"] == "NEEDS_REVIEW"
        assert mapping["unmapped_labels"] == [original_fields["attack_domain"]]
        assert any("injected task" in reason for reason in mapping["review_reasons"])
        assert any("NOASSERTION" in reason for reason in mapping["review_reasons"])
        assert validate_record(record) == []


def test_rejects_categories_that_do_not_contain_exactly_five_strings(tmp_path: Path) -> None:
    invalid = tmp_path / "text_attack_train.json"
    invalid.write_text(json.dumps({"Information Retrieval": ["one", "two"]}), encoding="utf-8")
    inputs = _inputs()
    inputs["text_attack_train"] = invalid

    with pytest.raises(ImporterError, match="exactly five payloads"):
        list(_importer().records(inputs))
