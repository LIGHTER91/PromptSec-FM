"""Deterministic, execution-free snapshots of pinned AgentDojo task definitions.

The acquisition step imports AgentDojo's public ``get_suites`` API once and writes a
small canonical JSON document. Ordinary dataset builds consume only that document;
they never instantiate an attack, run a benchmark, execute a tool, or call a model API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from promptsec.data.config import SourceConfig

SNAPSHOT_SCHEMA_VERSION = "0.1"
AGENTDOJO_PACKAGE = "agentdojo"
AGENTDOJO_PACKAGE_VERSION = "0.1.35"
AGENTDOJO_BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_REVISION = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
AGENTDOJO_WHEEL = "agentdojo-0.1.35-py3-none-any.whl"
AGENTDOJO_WHEEL_SHA256 = "364bea4219716b716bf639f504d195943f7f6a5535d312ca41d7098704a2affd"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_TASK_ID_RE = re.compile(r"^(user_task|injection_task)_(\d+)$")
_DIFFICULTIES = {"EASY", "MEDIUM", "HARD"}


class AgentDojoSnapshotError(ValueError):
    """Raised when AgentDojo definitions cannot be exported or verified safely."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a snapshot deterministically, including one final newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def snapshot_path(config: SourceConfig, artifact_path: str | Path) -> Path:
    """Resolve the configured snapshot next to a wheel, cache directory, or JSON fixture."""

    filename = config.acquisition.snapshot_filename
    if not filename:
        raise AgentDojoSnapshotError("AgentDojo acquisition has no snapshot_filename")
    candidate = Path(artifact_path)
    if candidate.is_dir():
        return candidate / filename
    if candidate.name == filename or candidate.suffix.lower() == ".json":
        return candidate
    return candidate.parent / filename


def export_snapshot(
    output: str | Path,
    package_version: str,
    revision: str,
    benchmark_version: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export installed AgentDojo definitions through its public ``get_suites`` API.

    AgentDojo must already be installed in the current interpreter. Importing the
    package loads its Python definitions but this function calls only ``get_suites``.
    """

    if package_version != AGENTDOJO_PACKAGE_VERSION:
        raise AgentDojoSnapshotError(
            f"unsupported AgentDojo package version {package_version!r}; "
            f"expected {AGENTDOJO_PACKAGE_VERSION!r}"
        )
    try:
        installed_version = metadata.version(AGENTDOJO_PACKAGE)
    except metadata.PackageNotFoundError as exc:
        raise AgentDojoSnapshotError(
            f"{AGENTDOJO_PACKAGE}=={package_version} is not installed in this interpreter"
        ) from exc
    if installed_version != package_version:
        raise AgentDojoSnapshotError(
            f"installed {AGENTDOJO_PACKAGE} version {installed_version!r} does not match "
            f"the pin {package_version!r}"
        )

    try:
        from agentdojo.task_suite import get_suites
    except (ImportError, AttributeError) as exc:
        raise AgentDojoSnapshotError(
            "cannot import the pinned AgentDojo public get_suites API"
        ) from exc

    try:
        suites = get_suites(benchmark_version)
    except Exception as exc:
        raise AgentDojoSnapshotError(
            f"AgentDojo get_suites({benchmark_version!r}) failed: {exc}"
        ) from exc
    snapshot = build_snapshot(
        suites,
        package_version=package_version,
        revision=revision,
        benchmark_version=benchmark_version,
    )
    write_snapshot(output, snapshot, overwrite=overwrite)
    return snapshot


def build_snapshot(
    suites: Mapping[str, Any],
    *,
    package_version: str = AGENTDOJO_PACKAGE_VERSION,
    revision: str = AGENTDOJO_REVISION,
    benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION,
) -> dict[str, Any]:
    """Build a snapshot from suite objects without invoking task or tool methods."""

    if not isinstance(suites, Mapping) or not suites:
        raise AgentDojoSnapshotError("get_suites returned no suites")

    serialized_suites: list[dict[str, Any]] = []
    for suite_id in sorted(suites):
        suite = suites[suite_id]
        if not isinstance(suite_id, str) or not suite_id:
            raise AgentDojoSnapshotError("suite identifiers must be non-empty strings")
        suite_name = _non_empty_string(getattr(suite, "name", None), f"suite {suite_id}.name")
        if suite_name != suite_id:
            raise AgentDojoSnapshotError(
                f"suite registry key {suite_id!r} does not match suite.name {suite_name!r}"
            )
        effective_version = _format_benchmark_version(
            getattr(suite, "benchmark_version", None), f"suite {suite_id}.benchmark_version"
        )
        tools = sorted(
            _non_empty_string(getattr(tool, "name", None), f"suite {suite_id}.tool.name")
            for tool in _as_sequence(getattr(suite, "tools", None), f"suite {suite_id}.tools")
        )
        user_tasks = _serialize_tasks(
            getattr(suite, "user_tasks", None), suite_id=suite_id, kind="user"
        )
        injection_tasks = _serialize_tasks(
            getattr(suite, "injection_tasks", None), suite_id=suite_id, kind="injection"
        )
        serialized_suites.append(
            {
                "effective_benchmark_version": effective_version,
                "initial_state_reference": (f"agentdojo/data/suites/{suite_id}/environment.yaml"),
                "injection_tasks": injection_tasks,
                "injection_vectors_reference": (
                    f"agentdojo/data/suites/{suite_id}/injection_vectors.yaml"
                ),
                "suite_id": suite_id,
                "tool_names": tools,
                "user_tasks": user_tasks,
            }
        )

    counts = _compute_counts(serialized_suites)
    snapshot: dict[str, Any] = {
        "counts": counts,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": {
            "benchmark_version": benchmark_version,
            "distribution_filename": AGENTDOJO_WHEEL,
            "distribution_sha256": AGENTDOJO_WHEEL_SHA256,
            "package_name": AGENTDOJO_PACKAGE,
            "package_version": package_version,
            "resolved_revision": revision,
        },
        "suites": serialized_suites,
    }
    _validate_snapshot_document(
        snapshot,
        expected_package_version=package_version,
        expected_benchmark_version=benchmark_version,
        expected_revision=revision,
    )
    return snapshot


def write_snapshot(
    output: str | Path,
    snapshot: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write canonical snapshot bytes and return their SHA-256 digest."""

    _validate_snapshot_document(snapshot)
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise AgentDojoSnapshotError(f"snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(snapshot)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def load_snapshot(
    path: str | Path,
    *,
    expected_package_version: str = AGENTDOJO_PACKAGE_VERSION,
    expected_benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION,
    expected_revision: str = AGENTDOJO_REVISION,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a canonical snapshot and enforce all configured pins."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise AgentDojoSnapshotError(f"cannot read AgentDojo snapshot {source}: {exc}") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise AgentDojoSnapshotError(
            f"AgentDojo snapshot checksum {actual_sha256} does not match {expected_sha256.lower()}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentDojoSnapshotError(f"invalid AgentDojo snapshot {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentDojoSnapshotError("AgentDojo snapshot root must be an object")
    _validate_snapshot_document(
        value,
        expected_package_version=expected_package_version,
        expected_benchmark_version=expected_benchmark_version,
        expected_revision=expected_revision,
    )
    if raw != canonical_json_bytes(value):
        raise AgentDojoSnapshotError(
            "AgentDojo snapshot is not in canonical sorted JSON form with one final newline"
        )
    return value


def validate_snapshot(path: str | Path, config: SourceConfig) -> dict[str, Any]:
    """Validate a snapshot against an AgentDojo source configuration."""

    acquisition = config.acquisition
    package_version = acquisition.package_version
    benchmark_version = acquisition.benchmark_version
    if not package_version or not benchmark_version or not config.revision:
        raise AgentDojoSnapshotError(
            "AgentDojo config must pin package_version, benchmark_version, and revision"
        )
    return load_snapshot(
        path,
        expected_package_version=package_version,
        expected_benchmark_version=benchmark_version,
        expected_revision=config.revision,
        expected_sha256=acquisition.snapshot_sha256,
    )


def iter_security_cases(snapshot: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one static case per suite/user-task/injection-task combination."""

    _validate_snapshot_document(snapshot)
    source = snapshot["source"]
    benchmark_version = source["benchmark_version"]
    for suite in snapshot["suites"]:
        for user_task in suite["user_tasks"]:
            for injection_task in suite["injection_tasks"]:
                suite_id = suite["suite_id"]
                user_task_id = user_task["id"]
                injection_task_id = injection_task["id"]
                security_case_id = (
                    f"agentdojo:{benchmark_version}:{suite_id}:{user_task_id}:{injection_task_id}"
                )
                yield {
                    "benchmark_version": benchmark_version,
                    "effective_benchmark_version": suite["effective_benchmark_version"],
                    "initial_state_reference": suite["initial_state_reference"],
                    "injection_goal": injection_task["goal"],
                    "injection_task_definition_class": injection_task["definition_class"],
                    "injection_task_definition_module": injection_task["definition_module"],
                    "injection_task_difficulty": injection_task["difficulty"],
                    "injection_task_id": injection_task_id,
                    "injection_vectors_reference": suite["injection_vectors_reference"],
                    "security_case_id": security_case_id,
                    "suite_id": suite_id,
                    "suite_tool_names": list(suite["tool_names"]),
                    "user_prompt": user_task["prompt"],
                    "user_task_definition_class": user_task["definition_class"],
                    "user_task_definition_module": user_task["definition_module"],
                    "user_task_difficulty": user_task["difficulty"],
                    "user_task_id": user_task_id,
                }


def snapshot_sha256(path: str | Path) -> str:
    """Return the SHA-256 of snapshot bytes."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise AgentDojoSnapshotError(f"cannot hash AgentDojo snapshot {path}: {exc}") from exc
    return digest.hexdigest()


def _serialize_tasks(
    tasks: Any,
    *,
    suite_id: str,
    kind: str,
) -> list[dict[str, str]]:
    if not isinstance(tasks, Mapping) or not tasks:
        raise AgentDojoSnapshotError(f"suite {suite_id}.{kind}_tasks must be a non-empty mapping")
    expected_prefix = "user_task" if kind == "user" else "injection_task"
    text_attribute = "PROMPT" if kind == "user" else "GOAL"
    text_key = "prompt" if kind == "user" else "goal"
    serialized: list[dict[str, str]] = []
    for task_id in sorted(tasks, key=lambda value: _task_sort_key(value, expected_prefix)):
        task = tasks[task_id]
        if not isinstance(task_id, str):
            raise AgentDojoSnapshotError(f"suite {suite_id} has a non-string {kind} task id")
        object_id = _non_empty_string(getattr(task, "ID", None), f"{suite_id}.{task_id}.ID")
        if object_id != task_id:
            raise AgentDojoSnapshotError(
                f"task registry key {task_id!r} does not match task.ID {object_id!r}"
            )
        _task_sort_key(task_id, expected_prefix)
        difficulty = getattr(task, "DIFFICULTY", None)
        difficulty_name = getattr(difficulty, "name", difficulty)
        difficulty_value = _non_empty_string(difficulty_name, f"{suite_id}.{task_id}.DIFFICULTY")
        if difficulty_value not in _DIFFICULTIES:
            raise AgentDojoSnapshotError(
                f"{suite_id}.{task_id}.DIFFICULTY has unsupported value {difficulty_value!r}"
            )
        serialized.append(
            {
                "definition_class": _non_empty_string(
                    getattr(type(task), "__name__", None), f"{suite_id}.{task_id}.class"
                ),
                "definition_module": _non_empty_string(
                    getattr(type(task), "__module__", None), f"{suite_id}.{task_id}.module"
                ),
                "difficulty": difficulty_value,
                "id": task_id,
                text_key: _non_empty_string(
                    getattr(task, text_attribute, None),
                    f"{suite_id}.{task_id}.{text_attribute}",
                ),
            }
        )
    return serialized


def _validate_snapshot_document(
    snapshot: Mapping[str, Any],
    *,
    expected_package_version: str | None = None,
    expected_benchmark_version: str | None = None,
    expected_revision: str | None = None,
) -> None:
    root = _object(snapshot, "snapshot")
    _exact_keys(root, {"counts", "snapshot_schema_version", "source", "suites"}, "snapshot")
    if root["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise AgentDojoSnapshotError(
            f"unsupported snapshot_schema_version {root['snapshot_schema_version']!r}"
        )
    source = _object(root["source"], "snapshot.source")
    _exact_keys(
        source,
        {
            "benchmark_version",
            "distribution_filename",
            "distribution_sha256",
            "package_name",
            "package_version",
            "resolved_revision",
        },
        "snapshot.source",
    )
    if source["package_name"] != AGENTDOJO_PACKAGE:
        raise AgentDojoSnapshotError("snapshot.source.package_name must be 'agentdojo'")
    package_version = _non_empty_string(
        source["package_version"], "snapshot.source.package_version"
    )
    benchmark_version = _non_empty_string(
        source["benchmark_version"], "snapshot.source.benchmark_version"
    )
    revision = _non_empty_string(source["resolved_revision"], "snapshot.source.resolved_revision")
    if not _VERSION_RE.fullmatch(benchmark_version):
        raise AgentDojoSnapshotError("snapshot.source.benchmark_version must be vMAJOR.MINOR.PATCH")
    if not _REVISION_RE.fullmatch(revision):
        raise AgentDojoSnapshotError(
            "snapshot.source.resolved_revision must be a lowercase Git SHA"
        )
    if source["distribution_filename"] != AGENTDOJO_WHEEL:
        raise AgentDojoSnapshotError("snapshot references an unexpected AgentDojo distribution")
    if source["distribution_sha256"] != AGENTDOJO_WHEEL_SHA256:
        raise AgentDojoSnapshotError("snapshot references an unexpected AgentDojo wheel checksum")
    if expected_package_version and package_version != expected_package_version:
        raise AgentDojoSnapshotError(
            f"snapshot package version {package_version!r} does not match "
            f"{expected_package_version!r}"
        )
    if expected_benchmark_version and benchmark_version != expected_benchmark_version:
        raise AgentDojoSnapshotError(
            f"snapshot benchmark version {benchmark_version!r} does not match "
            f"{expected_benchmark_version!r}"
        )
    if expected_revision and revision != expected_revision.lower():
        raise AgentDojoSnapshotError(
            f"snapshot revision {revision!r} does not match {expected_revision.lower()!r}"
        )

    suites = root["suites"]
    if not isinstance(suites, list) or not suites:
        raise AgentDojoSnapshotError("snapshot.suites must be a non-empty array")
    suite_ids: list[str] = []
    for suite_index, raw_suite in enumerate(suites):
        path = f"snapshot.suites[{suite_index}]"
        suite = _object(raw_suite, path)
        _exact_keys(
            suite,
            {
                "effective_benchmark_version",
                "initial_state_reference",
                "injection_tasks",
                "injection_vectors_reference",
                "suite_id",
                "tool_names",
                "user_tasks",
            },
            path,
        )
        suite_id = _non_empty_string(suite["suite_id"], f"{path}.suite_id")
        suite_ids.append(suite_id)
        effective = _non_empty_string(
            suite["effective_benchmark_version"], f"{path}.effective_benchmark_version"
        )
        if not _VERSION_RE.fullmatch(effective):
            raise AgentDojoSnapshotError(f"{path}.effective_benchmark_version is malformed")
        _safe_reference(suite["initial_state_reference"], f"{path}.initial_state_reference")
        _safe_reference(suite["injection_vectors_reference"], f"{path}.injection_vectors_reference")
        _validate_string_list(suite["tool_names"], f"{path}.tool_names")
        _validate_task_documents(suite["user_tasks"], path=path, kind="user")
        _validate_task_documents(suite["injection_tasks"], path=path, kind="injection")
    if suite_ids != sorted(suite_ids) or len(suite_ids) != len(set(suite_ids)):
        raise AgentDojoSnapshotError("snapshot suites must be unique and sorted by suite_id")

    counts = _object(root["counts"], "snapshot.counts")
    expected_counts = _compute_counts(suites)
    _exact_keys(counts, set(expected_counts), "snapshot.counts")
    if counts != expected_counts:
        raise AgentDojoSnapshotError(
            f"snapshot counts {counts!r} do not match definitions {expected_counts!r}"
        )


def _validate_task_documents(tasks: Any, *, path: str, kind: str) -> None:
    task_path = f"{path}.{kind}_tasks"
    if not isinstance(tasks, list) or not tasks:
        raise AgentDojoSnapshotError(f"{task_path} must be a non-empty array")
    text_key = "prompt" if kind == "user" else "goal"
    prefix = "user_task" if kind == "user" else "injection_task"
    task_ids: list[str] = []
    for index, raw_task in enumerate(tasks):
        item_path = f"{task_path}[{index}]"
        task = _object(raw_task, item_path)
        _exact_keys(
            task,
            {"definition_class", "definition_module", "difficulty", "id", text_key},
            item_path,
        )
        task_id = _non_empty_string(task["id"], f"{item_path}.id")
        _task_sort_key(task_id, prefix)
        task_ids.append(task_id)
        _non_empty_string(task[text_key], f"{item_path}.{text_key}")
        _non_empty_string(task["definition_class"], f"{item_path}.definition_class")
        _non_empty_string(task["definition_module"], f"{item_path}.definition_module")
        if task["difficulty"] not in _DIFFICULTIES:
            raise AgentDojoSnapshotError(f"{item_path}.difficulty is unsupported")
    expected_order = sorted(task_ids, key=lambda value: _task_sort_key(value, prefix))
    if task_ids != expected_order or len(task_ids) != len(set(task_ids)):
        raise AgentDojoSnapshotError(f"{task_path} ids must be unique and naturally sorted")


def _compute_counts(suites: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "injection_tasks": sum(len(suite["injection_tasks"]) for suite in suites),
        "security_cases": sum(
            len(suite["user_tasks"]) * len(suite["injection_tasks"]) for suite in suites
        ),
        "suites": len(suites),
        "user_tasks": sum(len(suite["user_tasks"]) for suite in suites),
    }


def _format_benchmark_version(value: Any, context: str) -> str:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(not isinstance(item, int) or item < 0 for item in value)
    ):
        raise AgentDojoSnapshotError(f"{context} must be a three-integer version")
    return "v" + ".".join(str(item) for item in value)


def _task_sort_key(value: Any, prefix: str) -> tuple[int, str]:
    if not isinstance(value, str):
        raise AgentDojoSnapshotError(f"{prefix} identifier must be a string")
    match = _TASK_ID_RE.fullmatch(value)
    if match is None or match.group(1) != prefix:
        raise AgentDojoSnapshotError(f"invalid {prefix} identifier {value!r}")
    return int(match.group(2)), value


def _as_sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise AgentDojoSnapshotError(f"{context} must be a non-empty sequence")
    return list(value)


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentDojoSnapshotError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AgentDojoSnapshotError(
            f"{context} keys are invalid; missing={missing}, extra={extra}"
        )


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentDojoSnapshotError(f"{context} must be a non-empty string")
    return value


def _safe_reference(value: Any, context: str) -> None:
    reference = _non_empty_string(value, context)
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts or "\\" in reference:
        raise AgentDojoSnapshotError(f"{context} must be a safe POSIX package-relative path")


def _validate_string_list(value: Any, context: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise AgentDojoSnapshotError(f"{context} must be a non-empty string array")
    if value != sorted(value) or len(value) != len(set(value)):
        raise AgentDojoSnapshotError(f"{context} must contain unique sorted values")


__all__ = [
    "AGENTDOJO_BENCHMARK_VERSION",
    "AGENTDOJO_PACKAGE_VERSION",
    "AGENTDOJO_REVISION",
    "AGENTDOJO_WHEEL_SHA256",
    "AgentDojoSnapshotError",
    "build_snapshot",
    "canonical_json_bytes",
    "export_snapshot",
    "iter_security_cases",
    "load_snapshot",
    "snapshot_path",
    "snapshot_sha256",
    "validate_snapshot",
    "write_snapshot",
]
