from __future__ import annotations

from pathlib import Path

import pytest

from promptsec.policybench.schemas import (
    GENERATION_RESPONSE_SCHEMA,
    POLICY_CATALOG_SCHEMA,
    POLICYBENCH_RECORD_SCHEMA,
    SCENARIO_BLUEPRINT_SCHEMA,
    load_schema,
    validate_generation_response_schema,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "schema_name",
    [
        POLICY_CATALOG_SCHEMA,
        SCENARIO_BLUEPRINT_SCHEMA,
        GENERATION_RESPONSE_SCHEMA,
        POLICYBENCH_RECORD_SCHEMA,
    ],
)
def test_policybench_contract_schemas_are_valid_draft_2020_12(schema_name: str) -> None:
    schema = load_schema(schema_name)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(f"/schemas/{schema_name}")


def test_blueprint_annotation_enums_exactly_match_the_frozen_annotation_schema() -> None:
    blueprint = load_schema(SCENARIO_BLUEPRINT_SCHEMA)
    annotation = load_schema(REPOSITORY_ROOT / "schemas" / "promptsec-annotation-v1.schema.json")
    expected = blueprint["$defs"]["expectedAnnotations"]["properties"]
    canonical = annotation["properties"]["annotations"]["properties"]

    scalar_fields = (
        "instruction_presence",
        "instruction_presentation",
        "instruction_addressee",
        "user_goal_alignment",
        "protected_policy_alignment",
        "authority_status",
        "annotation_status",
    )
    array_fields = ("attack_families", "attack_objectives")
    for field in scalar_fields:
        assert expected[field]["enum"] == canonical[field]["enum"]
    for field in array_fields:
        assert expected[field]["items"]["enum"] == canonical[field]["items"]["enum"]
    assert (
        expected["prompt_injection_verdict"]["enum"]
        == annotation["properties"]["derived"]["properties"]["prompt_injection_verdict"]["enum"]
    )


def test_generation_response_accepts_language_only_and_rejects_free_labels() -> None:
    response = {
        "schema_version": "0.1",
        "scenario_id": "banking_000001",
        "language": "fr",
        "protected_policy": "Confirmer tout virement.",
        "user_goal": "Consulter le solde.",
        "candidate_text": "Le solde affiché est de 20 €.",
        "directive_anchors": [],
        "injection_payload_anchors": [],
        "authority_claim_anchors": [],
    }

    assert validate_generation_response_schema(response) == []
    response["expected_annotations"] = {"instruction_presence": "NO_INSTRUCTION"}
    assert any(
        "expected_annotations" in error for error in validate_generation_response_schema(response)
    )


def test_record_profile_preserves_canonical_metadata_and_forbids_automatic_gold() -> None:
    schema = load_schema(POLICYBENCH_RECORD_SCHEMA)
    canonical_layer, policybench_layer = schema["allOf"]
    extension = schema["$defs"]["policybenchExtension"]

    assert canonical_layer["$ref"].endswith("/promptsec-dataset-record-v0.1.schema.json")
    assert "metadata" not in policybench_layer["properties"]
    assert extension["additionalProperties"] is False
    assert set(extension["properties"]["human_validation_status"]["enum"]) == {
        "PENDING",
        "IN_REVIEW",
        "CONFIRMED",
        "REJECTED",
    }
    quality_values = set(extension["properties"]["data_quality"]["enum"])
    assert quality_values == {
        "SILVER_TEMPLATE",
        "SILVER_GENERATED",
        "SILVER_VALIDATED",
        "GOLD_HUMAN_CONFIRMED",
        "EXCLUDED",
    }
    assert "GOLD_HUMAN_CONFIRMED" in quality_values


def test_catalogue_and_blueprint_share_one_action_vocabulary() -> None:
    catalogue = load_schema(POLICY_CATALOG_SCHEMA)
    blueprint = load_schema(SCENARIO_BLUEPRINT_SCHEMA)

    assert catalogue["$defs"]["action"]["enum"] == blueprint["$defs"]["action"]["enum"]
