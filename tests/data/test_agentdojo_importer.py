from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from promptsec.data.agentdojo_snapshot import canonical_json_bytes
from promptsec.data.config import SourceConfig
from promptsec.data.importers.agentdojo import AgentDojoImporter
from promptsec.data.importers.base import ImporterError
from promptsec.data.validation import validate_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "data"
    / "fixtures"
    / "agentdojo"
    / "agentdojo-0.1.35-v1.2.2.snapshot.json"
)


def _config() -> SourceConfig:
    config = SourceConfig.load(REPOSITORY_ROOT / "configs" / "sources" / "agentdojo.toml")
    return replace(
        config,
        acquisition=replace(config.acquisition, snapshot_sha256=None),
    )


def _importer() -> AgentDojoImporter:
    return AgentDojoImporter(_config(), imported_at="2026-07-11T12:00:00Z")


def _inputs(path: Path = FIXTURE) -> dict[str, Path]:
    return {"agentdojo_wheel": path}


def test_imports_one_record_per_static_security_pair_deterministically() -> None:
    first = list(_importer().records(_inputs()))
    second = list(_importer().records(_inputs()))

    assert len(first) == 5
    assert first == second
    assert len({record["id"] for record in first}) == 5
    source_ids = {
        record["metadata"]["dataset_provenance"]["source_record"]["id"] for record in first
    }
    assert len(source_ids) == 5
    assert "agentdojo:v1.2.2:workspace:user_task_1:injection_task_1" in source_ids


def test_preserves_agentic_provenance_and_maps_only_supported_axes() -> None:
    records = list(_importer().records(_inputs()))
    record = records[0]
    extensions = record["extensions"]
    agentic = extensions["agentic_source"]
    evidence = extensions["mapping_evidence"]

    assert record["context"]["user_goal"] == "Report the balance of the synthetic account."
    assert record["context"]["available_capabilities"] == [
        "get_balance",
        "schedule_transfer",
    ]
    assert record["content"]["delivery_mode"] == "INDIRECT"
    assert record["content"]["source_role"] == "TOOL"
    assert record["content"]["content_origin"] == "TOOL_OUTPUT"
    assert record["content"]["ingestion_path"] == "TOOL_CALL"
    assert record["annotations"]["attack_families"] == ["PROMPT_INJECTION"]
    assert record["annotations"]["attack_objectives"] == []
    assert record["annotations"]["authority_status"] == "UNKNOWN"
    assert record["annotations"]["user_goal_alignment"] == "UNDETERMINED"
    assert record["annotations"]["protected_policy_alignment"] == "UNDETERMINED"
    assert record["annotations"]["spans"] == [
        {
            "start": 0,
            "end": len(record["content"]["text"]),
            "type": "DIRECTIVE",
        }
    ]
    assert agentic["security_case_id_origin"] == "derived_from_upstream_task_pair"
    assert agentic["injection_bearing_content_status"] == ("not_materialized_static_definition")
    assert agentic["attack_name"] is None
    assert agentic["attack_configuration"] is None
    assert agentic["runtime_observations_status"] == "not_imported"
    assert agentic["targeted_tool_names"] == []
    assert evidence == {
        "label_origin": "SOURCE_DERIVED",
        "mapping_confidence": 0.6,
        "mapping_version": "agentdojo_mapping_v0.1",
        "rationale": [
            "SOURCE_DEFINES_INJECTION_TASK_GOAL",
            "STATIC_GOAL_IS_NOT_A_MATERIALIZED_ATTACK_PAYLOAD",
            "NO_RUNTIME_RESULT_USED_AS_GROUND_TRUTH",
            "AUTHORITY_AND_ALIGNMENT_REQUIRE_MANUAL_REVIEW",
        ],
        "requires_manual_review": True,
    }

    for candidate in records:
        provenance = candidate["metadata"]["dataset_provenance"]
        assert provenance["mapping"]["status"] == "NEEDS_REVIEW"
        assert provenance["source_record"]["original_fields"]["security_case_id"]
        assert provenance["source_record"]["original_labels"]["suite_id"]
        assert validate_record(candidate) == []


def test_related_injection_goals_share_template_family_without_execution_duplicates() -> None:
    records = list(_importer().records(_inputs()))
    workspace_goal_zero = [
        record
        for record in records
        if record["extensions"]["agentic_source"]["suite_id"] == "workspace"
        and record["extensions"]["agentic_source"]["injection_task_id"] == "injection_task_0"
    ]

    assert len(workspace_goal_zero) == 2
    assert (
        len(
            {
                record["extensions"]["agentic_source"]["template_family"]
                for record in workspace_goal_zero
            }
        )
        == 1
    )
    assert all(record["extensions"]["agentic_source"]["attack_name"] is None for record in records)


def test_resolves_snapshot_adjacent_to_downloaded_wheel(tmp_path: Path) -> None:
    config = _config()
    snapshot_name = config.acquisition.snapshot_filename
    assert snapshot_name is not None
    adjacent = tmp_path / snapshot_name
    adjacent.write_bytes(FIXTURE.read_bytes())

    records = list(_importer().records(_inputs(tmp_path / "agentdojo-0.1.35.whl")))

    assert len(records) == 5


def test_importer_never_imports_agentdojo_or_requires_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "agentdojo", None)

    assert len(list(_importer().records(_inputs()))) == 5


def test_rejects_missing_inputs_and_snapshot_pin_drift(tmp_path: Path) -> None:
    with pytest.raises(ImporterError, match="missing input"):
        list(_importer().records({}))
    with pytest.raises(ImporterError, match="unknown input"):
        list(_importer().records({"agentdojo_wheel": FIXTURE, "extra": FIXTURE}))

    snapshot = copy.deepcopy(__import__("json").loads(FIXTURE.read_text(encoding="utf-8")))
    snapshot["source"]["package_version"] = "0.1.34"
    bad = tmp_path / "bad.json"
    bad.write_bytes(canonical_json_bytes(snapshot))
    with pytest.raises(ImporterError, match="package version"):
        list(_importer().records(_inputs(bad)))
