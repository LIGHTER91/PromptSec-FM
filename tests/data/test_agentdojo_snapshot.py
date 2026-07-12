from __future__ import annotations

import copy
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from promptsec.data import agentdojo_snapshot as snapshot_module
from promptsec.data.agentdojo_snapshot import (
    AGENTDOJO_REVISION,
    AgentDojoSnapshotError,
    build_snapshot,
    canonical_json_bytes,
    export_snapshot,
    iter_security_cases,
    load_snapshot,
    snapshot_path,
    write_snapshot,
)
from promptsec.data.config import SourceConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "data"
    / "fixtures"
    / "agentdojo"
    / "agentdojo-0.1.35-v1.2.2.snapshot.json"
)


class _Difficulty(Enum):
    EASY = 1
    MEDIUM = 2


class _UserTask:
    ID = "user_task_0"
    PROMPT = "Summarize a synthetic note."
    DIFFICULTY = _Difficulty.EASY

    def ground_truth(self, *_: object) -> None:
        raise AssertionError("ground_truth must not be called")


class _InjectionTask:
    ID = "injection_task_0"
    GOAL = "Send a synthetic token to an external test address."
    DIFFICULTY = _Difficulty.MEDIUM

    def ground_truth(self, *_: object) -> None:
        raise AssertionError("ground_truth must not be called")


def _fake_suites() -> dict[str, SimpleNamespace]:
    return {
        "synthetic": SimpleNamespace(
            name="synthetic",
            benchmark_version=(1, 2, 2),
            tools=[SimpleNamespace(name="read_note")],
            user_tasks={"user_task_0": _UserTask()},
            injection_tasks={"injection_task_0": _InjectionTask()},
        )
    }


def test_loads_canonical_fixture_and_enumerates_each_pair_once() -> None:
    snapshot = load_snapshot(FIXTURE)
    cases = list(iter_security_cases(snapshot))

    assert snapshot["counts"] == {
        "injection_tasks": 3,
        "security_cases": 5,
        "suites": 2,
        "user_tasks": 3,
    }
    assert len(cases) == 5
    assert len({case["security_case_id"] for case in cases}) == 5
    assert cases[0]["security_case_id"] == ("agentdojo:v1.2.2:banking:user_task_0:injection_task_0")


def test_snapshot_serialization_is_byte_deterministic(tmp_path: Path) -> None:
    snapshot = load_snapshot(FIXTURE)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_hash = write_snapshot(first, snapshot)
    second_hash = write_snapshot(second, snapshot)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes() == canonical_json_bytes(snapshot)


def test_build_snapshot_reads_attributes_without_executing_tasks_or_tools() -> None:
    snapshot = build_snapshot(_fake_suites())

    assert snapshot["counts"]["security_cases"] == 1
    assert snapshot["suites"][0]["injection_tasks"][0]["goal"].startswith("Send")


def test_export_uses_only_installed_public_get_suites_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    package = types.ModuleType("agentdojo")
    package.__path__ = []  # type: ignore[attr-defined]
    task_suite = types.ModuleType("agentdojo.task_suite")

    def get_suites(version: str) -> dict[str, SimpleNamespace]:
        calls.append(version)
        return _fake_suites()

    task_suite.get_suites = get_suites  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentdojo", package)
    monkeypatch.setitem(sys.modules, "agentdojo.task_suite", task_suite)
    monkeypatch.setattr(snapshot_module.metadata, "version", lambda _: "0.1.35")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    output = tmp_path / "snapshot.json"

    result = export_snapshot(output, "0.1.35", AGENTDOJO_REVISION, "v1.2.2")

    assert calls == ["v1.2.2"]
    assert result["counts"]["security_cases"] == 1
    assert load_snapshot(output)["counts"]["security_cases"] == 1
    assert "agentdojo.benchmark" not in sys.modules


def test_export_rejects_adapter_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot_module.metadata, "version", lambda _: "0.1.34")

    with pytest.raises(AgentDojoSnapshotError, match="does not match the pin"):
        export_snapshot(tmp_path / "snapshot.json", "0.1.35", AGENTDOJO_REVISION, "v1.2.2")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("package_version", "0.1.34", "package version"),
        ("benchmark_version", "v1.2.1", "benchmark version"),
        ("resolved_revision", "0" * 40, "revision"),
    ],
)
def test_loader_rejects_pin_mismatches(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    snapshot["source"][field] = value
    path = tmp_path / "snapshot.json"
    path.write_bytes(canonical_json_bytes(snapshot))

    with pytest.raises(AgentDojoSnapshotError, match=message):
        load_snapshot(path)


def test_loader_rejects_checksum_and_noncanonical_json(tmp_path: Path) -> None:
    with pytest.raises(AgentDojoSnapshotError, match="checksum"):
        load_snapshot(FIXTURE, expected_sha256="0" * 64)

    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    with pytest.raises(AgentDojoSnapshotError, match="not in canonical"):
        load_snapshot(pretty)


def test_validator_rejects_counts_drift_extra_fields_and_unsorted_tasks() -> None:
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    bad_counts = copy.deepcopy(snapshot)
    bad_counts["counts"]["security_cases"] = 999
    with pytest.raises(AgentDojoSnapshotError, match="counts"):
        write_snapshot(Path("unused-counts.json"), bad_counts)

    extra = copy.deepcopy(snapshot)
    extra["unexpected"] = True
    with pytest.raises(AgentDojoSnapshotError, match="extra"):
        write_snapshot(Path("unused-extra.json"), extra)

    unsorted = copy.deepcopy(snapshot)
    unsorted["suites"][1]["user_tasks"].reverse()
    with pytest.raises(AgentDojoSnapshotError, match="naturally sorted"):
        write_snapshot(Path("unused-unsorted.json"), unsorted)


def test_snapshot_path_supports_cache_wheel_and_json_fixture(tmp_path: Path) -> None:
    config = SourceConfig.load(REPOSITORY_ROOT / "configs" / "sources" / "agentdojo.toml")
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = cache / config.acquisition.snapshot_filename  # type: ignore[operator]

    assert snapshot_path(config, cache) == expected
    assert snapshot_path(config, cache / "agentdojo.whl") == expected
    assert snapshot_path(config, FIXTURE) == FIXTURE
