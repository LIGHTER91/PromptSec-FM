"""Strict configuration for a reproducible PromptSec dataset release."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from promptsec.data.hashing import sha256_file


class ReleaseConfigError(ValueError):
    """Raised when a dataset release configuration is incomplete or unsafe."""


ANNOTATION_TIERS = {
    "GOLD_SOURCE",
    "DETERMINISTIC_MAPPING",
    "HEURISTIC_MAPPING",
    "UNANNOTATED",
}
SPLIT_NAMES = ("train", "validation", "test_id")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseConfigError(f"{context} must be a mapping")
    return value


def _required(table: dict[str, Any], key: str, context: str) -> Any:
    if key not in table:
        raise ReleaseConfigError(f"{context}: missing required key {key!r}")
    return table[key]


def _ratios(value: Any, context: str) -> dict[str, float]:
    table = _mapping(value, context)
    if set(table) != set(SPLIT_NAMES):
        raise ReleaseConfigError(f"{context} must contain exactly {list(SPLIT_NAMES)}")
    ratios = {name: _unit_float(table[name], f"{context}.{name}") for name in SPLIT_NAMES}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ReleaseConfigError(f"{context} ratios must sum to 1.0")
    return ratios


def _unit_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseConfigError(f"{context} must be a number within [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ReleaseConfigError(f"{context} must be a finite number within [0, 1]")
    return result


def _integer_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseConfigError("dataset.seed must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    id: str
    title: str
    taxonomy_version: str
    record_schema_version: str
    imported_at: str
    seed: int


@dataclass(frozen=True, slots=True)
class ReleasePaths:
    raw_dir: Path
    output: Path
    statistics_json: Path
    statistics_markdown: Path
    review_queue: Path
    agentic_review_queue: Path | None


@dataclass(frozen=True, slots=True)
class MappingQualityConfig:
    review_threshold: float
    profiles: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DeduplicationSettings:
    semantic_threshold: float
    variant_threshold: float


@dataclass(frozen=True, slots=True)
class SplitSettings:
    held_out_source: str
    held_out_family: str
    general_ratios: dict[str, float]
    notinject_ratios: dict[str, float]
    agentic_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetReleaseConfig:
    path: Path
    schema_version: str
    project_root: Path
    identity: DatasetIdentity
    paths: ReleasePaths
    source_configs: tuple[Path, ...]
    mapping_quality: MappingQualityConfig
    deduplication: DeduplicationSettings
    splits: SplitSettings

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def split_names(self) -> tuple[str, ...]:
        base = (
            "train",
            "validation",
            "test_id",
            "test_held_out_source",
            "test_held_out_family",
        )
        return (*base, "test_agentic_provisional") if self.splits.agentic_sources else base

    @classmethod
    def load(cls, path: str | Path) -> DatasetReleaseConfig:
        config_path = Path(path).resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ReleaseConfigError(f"cannot read {config_path}: {exc}") from exc
        root = _mapping(raw, "config")
        schema_version = str(_required(root, "schema_version", "config"))
        if schema_version not in {"0.1", "0.2"}:
            raise ReleaseConfigError("config.schema_version must equal '0.1' or '0.2'")

        project_root_value = str(_required(root, "project_root", "config"))
        project_root = (config_path.parent / project_root_value).resolve()
        dataset = _mapping(_required(root, "dataset", "config"), "dataset")
        identity = DatasetIdentity(
            id=str(_required(dataset, "id", "dataset")),
            title=str(_required(dataset, "title", "dataset")),
            taxonomy_version=str(_required(dataset, "taxonomy_version", "dataset")),
            record_schema_version=str(_required(dataset, "record_schema_version", "dataset")),
            imported_at=str(_required(dataset, "imported_at", "dataset")),
            seed=_integer_seed(_required(dataset, "seed", "dataset")),
        )
        if identity.taxonomy_version != "1.0" or identity.record_schema_version != "0.1":
            raise ReleaseConfigError("dataset must use taxonomy 1.0 and record schema 0.1")

        path_table = _mapping(_required(root, "paths", "config"), "paths")

        def resolve_path(key: str) -> Path:
            return (project_root / str(_required(path_table, key, "paths"))).resolve()

        paths = ReleasePaths(
            raw_dir=resolve_path("raw_dir"),
            output=resolve_path("output"),
            statistics_json=resolve_path("statistics_json"),
            statistics_markdown=resolve_path("statistics_markdown"),
            review_queue=resolve_path("review_queue"),
            agentic_review_queue=(
                resolve_path("agentic_review_queue")
                if "agentic_review_queue" in path_table
                else None
            ),
        )

        source_values = _required(root, "sources", "config")
        if not isinstance(source_values, list) or not source_values:
            raise ReleaseConfigError("config.sources must be a non-empty list")
        source_configs = tuple((project_root / str(value)).resolve() for value in source_values)
        if len(source_configs) != len(set(source_configs)):
            raise ReleaseConfigError("config.sources contains duplicate paths")

        mapping_table = _mapping(_required(root, "mapping_quality", "config"), "mapping_quality")
        threshold = _unit_float(
            _required(mapping_table, "review_threshold", "mapping_quality"),
            "mapping_quality.review_threshold",
        )
        raw_profiles = _mapping(
            _required(mapping_table, "profiles", "mapping_quality"),
            "mapping_quality.profiles",
        )
        profiles: dict[str, dict[str, Any]] = {}
        for source_id, profile_value in raw_profiles.items():
            profile = _mapping(profile_value, f"mapping_quality.profiles.{source_id}")
            tier = str(_required(profile, "annotation_tier", f"profile {source_id}"))
            confidence = _unit_float(
                _required(profile, "mapping_confidence", f"profile {source_id}"),
                f"profile {source_id}.mapping_confidence",
            )
            reasons = profile.get("review_reasons", [])
            if tier not in ANNOTATION_TIERS:
                raise ReleaseConfigError(f"profile {source_id}: invalid annotation_tier {tier}")
            if not isinstance(reasons, list) or not all(
                isinstance(reason, str) and reason for reason in reasons
            ):
                raise ReleaseConfigError(f"profile {source_id}: review_reasons must be strings")
            requires_manual_review = profile.get("requires_manual_review", False)
            if not isinstance(requires_manual_review, bool):
                raise ReleaseConfigError(
                    f"profile {source_id}: requires_manual_review must be a boolean"
                )
            profiles[str(source_id)] = {
                "annotation_tier": tier,
                "mapping_confidence": confidence,
                "requires_manual_review": requires_manual_review,
                "review_reasons": list(reasons),
            }
        mapping_quality = MappingQualityConfig(threshold, profiles)

        dedup_table = _mapping(_required(root, "deduplication", "config"), "deduplication")
        semantic_threshold = _unit_float(
            _required(dedup_table, "semantic_threshold", "deduplication"),
            "deduplication.semantic_threshold",
        )
        variant_threshold = _unit_float(
            _required(dedup_table, "variant_threshold", "deduplication"),
            "deduplication.variant_threshold",
        )
        if semantic_threshold > variant_threshold:
            raise ReleaseConfigError(
                "deduplication requires 0 <= semantic_threshold <= variant_threshold <= 1"
            )
        deduplication = DeduplicationSettings(semantic_threshold, variant_threshold)

        split_table = _mapping(_required(root, "splits", "config"), "splits")
        agentic_sources_value = split_table.get("agentic_sources", [])
        if not isinstance(agentic_sources_value, list) or not all(
            isinstance(source_id, str) and source_id for source_id in agentic_sources_value
        ):
            raise ReleaseConfigError("splits.agentic_sources must be a string array")
        if len(agentic_sources_value) != len(set(agentic_sources_value)):
            raise ReleaseConfigError("splits.agentic_sources contains duplicates")
        splits = SplitSettings(
            held_out_source=str(_required(split_table, "held_out_source", "splits")),
            held_out_family=str(_required(split_table, "held_out_family", "splits")),
            general_ratios=_ratios(
                _required(split_table, "general_ratios", "splits"),
                "splits.general_ratios",
            ),
            notinject_ratios=_ratios(
                _required(split_table, "notinject_ratios", "splits"),
                "splits.notinject_ratios",
            ),
            agentic_sources=tuple(sorted(agentic_sources_value)),
        )
        return cls(
            path=config_path,
            schema_version=schema_version,
            project_root=project_root,
            identity=identity,
            paths=paths,
            source_configs=source_configs,
            mapping_quality=mapping_quality,
            deduplication=deduplication,
            splits=splits,
        )


__all__ = [
    "DatasetReleaseConfig",
    "DeduplicationSettings",
    "MappingQualityConfig",
    "ReleaseConfigError",
    "SplitSettings",
]
