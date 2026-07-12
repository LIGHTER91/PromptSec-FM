"""Acquire immutable upstream sources for reproducible, offline dataset builds.

The acquisition layer deliberately separates network access from ordinary imports.
After this step succeeds, ``offline=True`` verifies and reuses only the pinned local
artifacts (and, when applicable, the deterministic AgentDojo definition snapshot).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from promptsec.data.agentdojo_snapshot import (
    AgentDojoSnapshotError,
    export_snapshot,
    snapshot_path,
    validate_snapshot,
)
from promptsec.data.config import SourceConfig
from promptsec.data.fetch import FetchError, fetch_artifacts

_ARTIFACT_ONLY_METHODS = {"http_artifacts", "pinned_raw_files"}
_SNAPSHOT_METHOD = "python_package_snapshot"
_SUPPORTED_METHODS = _ARTIFACT_ONLY_METHODS | {_SNAPSHOT_METHOD}


class AcquisitionError(RuntimeError):
    """Raised when an immutable source cannot be prepared or verified."""


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Verified local inputs for one configured source."""

    source_id: str
    artifacts: dict[str, Path]
    snapshot: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": {
                artifact_id: path.resolve().as_posix()
                for artifact_id, path in sorted(self.artifacts.items())
            },
            "snapshot": self.snapshot.resolve().as_posix() if self.snapshot else None,
        }


def acquire_source(
    config: SourceConfig,
    destination: str | Path,
    *,
    overwrite: bool = False,
    offline: bool = False,
    local_path: str | Path | None = None,
) -> AcquisitionResult:
    """Acquire and verify one pinned source.

    ``local_path`` is an explicit source-level override. It remains subject to every
    configured checksum or Git revision pin. ``offline`` never downloads an artifact
    and never exports a missing package snapshot.
    """

    method = config.acquisition.method
    if method not in _SUPPORTED_METHODS:
        raise AcquisitionError(
            f"source {config.id!r} uses unsupported acquisition method {method!r}"
        )
    if offline and overwrite:
        raise AcquisitionError("offline acquisition cannot use overwrite")

    try:
        artifacts = fetch_artifacts(
            config,
            destination,
            overwrite=overwrite,
            offline=offline,
            local_path=local_path,
        )
    except FetchError as exc:
        raise AcquisitionError(f"cannot acquire source {config.id!r}: {exc}") from exc

    snapshot: Path | None = None
    if method == _SNAPSHOT_METHOD:
        snapshot = _acquire_package_snapshot(
            config,
            artifacts,
            overwrite=overwrite,
            offline=offline,
        )
    return AcquisitionResult(source_id=config.id, artifacts=artifacts, snapshot=snapshot)


def acquire_sources(
    configs: Iterable[SourceConfig],
    destination: str | Path,
    *,
    source_ids: Iterable[str] | None = None,
    overwrite: bool = False,
    offline: bool = False,
    local_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, AcquisitionResult]:
    """Acquire a selected set of unique source configurations in stable order."""

    by_id: dict[str, SourceConfig] = {}
    for config in configs:
        if config.id in by_id:
            raise AcquisitionError(f"duplicate configured source id: {config.id!r}")
        by_id[config.id] = config

    selected = set(source_ids) if source_ids is not None else set(by_id)
    unknown = sorted(selected - set(by_id))
    if unknown:
        raise AcquisitionError(f"requested sources are not configured: {unknown}")
    overrides = dict(local_paths or {})
    unknown_overrides = sorted(set(overrides) - selected)
    if unknown_overrides:
        raise AcquisitionError(
            f"local source overrides do not match selected sources: {unknown_overrides}"
        )

    return {
        source_id: acquire_source(
            by_id[source_id],
            destination,
            overwrite=overwrite,
            offline=offline,
            local_path=overrides.get(source_id),
        )
        for source_id in sorted(selected)
    }


def _acquire_package_snapshot(
    config: SourceConfig,
    artifacts: Mapping[str, Path],
    *,
    overwrite: bool,
    offline: bool,
) -> Path:
    acquisition = config.acquisition
    if len(artifacts) != 1:
        raise AcquisitionError(
            f"source {config.id!r} package snapshot acquisition requires one artifact"
        )
    if not acquisition.package_version:
        raise AcquisitionError(f"source {config.id!r} has no pinned package_version")
    if not acquisition.benchmark_version:
        raise AcquisitionError(f"source {config.id!r} has no pinned benchmark_version")
    if not config.revision:
        raise AcquisitionError(f"source {config.id!r} has no pinned revision")

    package_artifact = next(iter(artifacts.values()))
    try:
        target = snapshot_path(config, package_artifact)
    except AgentDojoSnapshotError as exc:
        raise AcquisitionError(f"invalid snapshot configuration for {config.id!r}: {exc}") from exc

    if offline and not target.is_file():
        raise AcquisitionError(
            f"offline mode: missing pinned snapshot for source {config.id!r} at {target}"
        )

    if not offline and (overwrite or not target.is_file()):
        try:
            export_snapshot(
                target,
                acquisition.package_version,
                config.revision,
                acquisition.benchmark_version,
                overwrite=overwrite,
            )
        except AgentDojoSnapshotError as exc:
            raise AcquisitionError(
                f"cannot export the pinned snapshot for source {config.id!r}: {exc}. "
                "Install the exact optional acquisition dependency, then retry"
            ) from exc

    try:
        validate_snapshot(target, config)
    except AgentDojoSnapshotError as exc:
        raise AcquisitionError(
            f"snapshot verification failed for source {config.id!r}: {exc}"
        ) from exc
    return target


__all__ = [
    "AcquisitionError",
    "AcquisitionResult",
    "acquire_source",
    "acquire_sources",
]
