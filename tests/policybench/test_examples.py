from __future__ import annotations

import json
import shutil
from pathlib import Path

from promptsec.data.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)
from promptsec.policybench.generation import GenerationOptions, generate_policybench
from promptsec.policybench.io import iter_jsonl, read_json_object
from promptsec.policybench.schemas import (
    validate_generation_response_schema,
    validate_policybench_record_schema,
    validate_scenario_blueprint_schema,
)
from promptsec.policybench.validation import (
    iter_dataset_records,
    validate_policybench_record,
    validate_release_directory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "policybench"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "policybench_v0.1.yaml"
REPRESENTATIVE_SCENARIO_ID = "pb_email_000693"


def _load_json(path: Path) -> dict:
    return read_json_object(path)


def _checksum_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (EXAMPLE_ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    return entries


def _assert_no_blueprint_surface_leakage(
    blueprint: dict,
    realization: dict,
) -> None:
    surface = "\n".join(
        value
        for field in ("candidate_text", "user_goal", "protected_policy")
        if isinstance((value := realization.get(field)), str)
    ).casefold()
    assert blueprint["scenario_id"].casefold() not in surface
    assert blueprint["category"].casefold() not in surface
    assert "scenario_reference" not in surface


def test_checked_in_example_contracts_hashes_spans_and_silver_state() -> None:
    blueprint = _load_json(EXAMPLE_ROOT / "blueprint.json")
    response = _load_json(EXAMPLE_ROOT / "generation_response.json")
    record = _load_json(EXAMPLE_ROOT / "record.json")
    manifest = _load_json(EXAMPLE_ROOT / "accepted_artifact_manifest.json")

    assert validate_scenario_blueprint_schema(blueprint) == []
    assert validate_generation_response_schema(response) == []
    assert validate_policybench_record_schema(record) == []
    assert validate_policybench_record(record, blueprint) == []
    assert blueprint["scenario_id"] == response["scenario_id"] == REPRESENTATIVE_SCENARIO_ID

    extension = record["extensions"]["policybench_v0_1"]
    assert extension["data_quality"] == "SILVER_VALIDATED"
    assert extension["human_validation_status"] == "PENDING"
    assert record["annotations"]["annotator_confidence"] == 0.0
    assert "GOLD" not in repr(record)

    spans = record["annotations"]["spans"]
    assert spans == [
        {"end": 262, "start": 68, "type": "DIRECTIVE"},
        {"end": 262, "start": 68, "type": "INJECTION_PAYLOAD"},
    ]
    candidate = response["candidate_text"]
    for span in spans:
        assert candidate[span["start"] : span["end"]] == response["directive_anchors"][0]["text"]
    _assert_no_blueprint_surface_leakage(blueprint, response)

    checksum_entries = _checksum_entries()
    assert set(checksum_entries) == {
        "accepted/pb_email_000693.json",
        "accepted/pb_web_and_purchases_000158.json",
        "accepted_artifact_manifest.json",
        "blueprint.json",
        "generation_response.json",
        "record.json",
    }
    for name, expected_digest in checksum_entries.items():
        assert sha256_file(EXAMPLE_ROOT / name) == expected_digest

    manifest_entries = {item["scenario_id"]: item for item in manifest["records"]}
    assert set(manifest_entries) == {
        "pb_email_000693",
        "pb_web_and_purchases_000158",
    }
    for scenario_id, entry in manifest_entries.items():
        artifact = _load_json(EXAMPLE_ROOT / entry["path"])
        unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        assert artifact["scenario_id"] == scenario_id
        assert artifact["artifact_sha256"] == sha256_json(unsigned)
        assert entry["artifact_sha256"] == artifact["artifact_sha256"]
        raw_response = artifact["raw_response"]
        assert raw_response == canonical_json_bytes(artifact["realization"]).decode("utf-8")
        assert json.loads(raw_response) == artifact["realization"]
        assert artifact["generator_metadata"]["raw_generation_sha256"] == sha256_text(raw_response)
    assert manifest_entries[REPRESENTATIVE_SCENARIO_ID]["path"].startswith("accepted/")
    representative_artifact = _load_json(
        EXAMPLE_ROOT / manifest_entries[REPRESENTATIVE_SCENARIO_ID]["path"]
    )
    assert representative_artifact["realization"] == response


def test_documented_fixture_replays_offline_without_invoking_a_provider(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "src").mkdir()
    shutil.copyfile(REPOSITORY_ROOT / "pyproject.toml", checkout / "pyproject.toml")
    shutil.copyfile(CONFIG_PATH, checkout / "configs" / CONFIG_PATH.name)
    shutil.copytree(
        REPOSITORY_ROOT / "data" / "policybench" / "policies",
        checkout / "data" / "policybench" / "policies",
    )
    shutil.copytree(
        REPOSITORY_ROOT / "prompts" / "policybench",
        checkout / "prompts" / "policybench",
    )
    accepted_root = checkout / "data" / "generated" / "accepted" / "policybench-v0.1"
    shutil.copytree(EXAMPLE_ROOT / "accepted", accepted_root)

    output = checkout / "data" / "generated" / "policybench-example-v0.1"
    result = generate_policybench(
        checkout / "configs" / CONFIG_PATH.name,
        output=output,
        options=GenerationOptions(
            offline=True,
            max_records=2,
            provider="mock",
            model="deterministic-template-v1",
            seed=20260715,
            concurrency=1,
        ),
    )

    assert result.records == result.reused_accepted_artifacts == 2
    assert result.automatic_gold_records == 0
    records = iter_dataset_records(output)
    assert validate_release_directory(output, records)["validation_status"] == "PASS"
    assert all(
        record["extensions"]["policybench_v0_1"]["data_quality"] == "SILVER_VALIDATED"
        for record in records
    )
    assert all(
        record["extensions"]["policybench_v0_1"]["human_validation_status"] == "PENDING"
        for record in records
    )

    representative = _load_json(EXAMPLE_ROOT / "record.json")
    assert representative in records
    output_manifest = _load_json(output / "accepted_artifact_manifest.json")
    fixture_manifest = _load_json(EXAMPLE_ROOT / "accepted_artifact_manifest.json")
    assert output_manifest["acquisition_fingerprint"] == fixture_manifest["acquisition_fingerprint"]
    assert output_manifest["config_sha256"] == fixture_manifest["config_sha256"]
    assert output_manifest["records"] == [
        {**item, "path": Path(item["path"]).name} for item in fixture_manifest["records"]
    ]

    blueprints = {item["scenario_id"]: item for item in iter_jsonl(output / "blueprints.jsonl")}
    assert set(blueprints) == {
        "pb_email_000693",
        "pb_web_and_purchases_000158",
    }
    for scenario_id, blueprint in blueprints.items():
        artifact = _load_json(accepted_root / f"{scenario_id}.json")
        _assert_no_blueprint_surface_leakage(blueprint, artifact["realization"])

    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "--offline --provider mock --model deterministic-template-v1" in readme
    assert "sha256sum -c checksums.sha256" in readme
