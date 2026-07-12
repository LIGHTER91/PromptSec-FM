from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from promptsec.data.config import ConfigError, SourceConfig

ROOT = Path(__file__).resolve().parents[2]


def test_all_six_source_configs_are_pinned_and_linked_to_manifests() -> None:
    paths = sorted((ROOT / "configs" / "sources").glob("*.toml"))
    configs = [SourceConfig.load(path) for path in paths]

    assert {config.id for config in configs} == {
        "agentdojo",
        "bipia",
        "injecagent",
        "notinject",
        "open_prompt_injection",
        "promptinject",
    }
    for config in configs:
        assert config.revision
        assert len(config.revision) == 40
        assert config.importer.startswith("promptsec.data.importers.")
        assert (ROOT / config.license_manifest).is_file()
        assert all(artifact.sha256 for artifact in config.artifacts)
        assert all(
            config.revision in artifact.url
            for artifact in config.artifacts
            if config.id not in {"agentdojo", "notinject"}
        )

    agentdojo = next(config for config in configs if config.id == "agentdojo")
    assert agentdojo.version == "0.1.35"
    assert agentdojo.acquisition.package_name == "agentdojo"
    assert agentdojo.acquisition.package_version == "0.1.35"
    assert agentdojo.acquisition.benchmark_version == "v1.2.2"
    assert agentdojo.artifacts[0].sha256 == (
        "364bea4219716b716bf639f504d195943f7f6a5535d312ca41d7098704a2affd"
    )


def test_source_lock_matches_the_six_source_configs() -> None:
    lock = json.loads((ROOT / "manifests" / "source-lock.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in lock["sources"]}
    configs = {
        config.id: config
        for config in (
            SourceConfig.load(path)
            for path in sorted((ROOT / "configs" / "sources").glob("*.toml"))
        )
    }

    assert lock["schema_version"] == "0.1"
    assert entries.keys() == configs.keys()
    for source_id, config in configs.items():
        entry = entries[source_id]
        assert entry["config"] == f"configs/sources/{source_id}.toml"
        assert entry["version"] == config.version
        assert entry["revision"] == config.revision
        assert entry["artifacts"] == {
            artifact.id: artifact.sha256
            for artifact in sorted(config.artifacts, key=lambda artifact: artifact.id)
        }


def test_license_manifests_validate_and_match_configs() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "source-license-manifest-v0.1.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    for config_path in sorted((ROOT / "configs" / "sources").glob("*.toml")):
        config = SourceConfig.load(config_path)
        manifest = json.loads((ROOT / config.license_manifest).read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(manifest), key=lambda error: error.json_path)
        assert not errors, [f"{error.json_path}: {error.message}" for error in errors]
        assert manifest["source_id"] == config.id


def test_bipia_manifest_does_not_flatten_dataset_licenses() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "sources" / "bipia.json").read_text(encoding="utf-8")
    )

    assert manifest["license_status"] == "multiple"
    expressions = {component["license_expression"] for component in manifest["components"]}
    assert {"MIT", "CC-BY-SA-4.0", "NOASSERTION"} <= expressions


def test_source_field_lists_cannot_be_empty(tmp_path) -> None:
    source = (ROOT / "configs" / "sources" / "promptinject.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace('text = ["instruction"]', "text = []"), encoding="utf-8")

    with pytest.raises(ConfigError, match="fields.text"):
        SourceConfig.load(path)
