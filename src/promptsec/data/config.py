"""Strict TOML source configuration for dataset importers."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file


class ConfigError(ValueError):
    """Raised when a source configuration is malformed."""


SUPPORTED_FORMATS = {"csv", "tsv", "json", "jsonl", "python", "text", "zip"}


def _required(table: dict[str, Any], key: str, context: str) -> Any:
    if key not in table:
        raise ConfigError(f"{context}: missing required key {key!r}")
    return table[key]


def _tuple_of_strings(value: Any, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"{context} must be a non-empty string array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ScenarioDefaults:
    language: str
    delivery_mode: str
    source_role: str
    content_origin: str
    ingestion_path: str
    modality: str
    source_integrity: str


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    id: str
    split: str
    format: str
    url: str
    sha256: str | None = None
    records_path: str | None = None
    encoding: str = "utf-8"


@dataclass(frozen=True, slots=True)
class FieldConfig:
    text: tuple[str, ...]
    labels: tuple[str, ...]
    record_id: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    path: Path
    schema_version: str
    id: str
    name: str
    homepage: str
    repository: str
    version: str | None
    revision: str | None
    license_manifest: str
    importer: str
    scenario: ScenarioDefaults
    fields: FieldConfig
    artifacts: tuple[ArtifactConfig, ...]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @classmethod
    def load(cls, path: str | Path) -> SourceConfig:
        config_path = Path(path)
        try:
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read {config_path}: {exc}") from exc

        schema_version = _required(raw, "schema_version", "config")
        if schema_version != "0.1":
            raise ConfigError(f"config: unsupported schema_version {schema_version!r}")

        source = _required(raw, "source", "config")
        scenario_raw = _required(raw, "scenario", "config")
        fields_raw = _required(raw, "fields", "config")
        artifacts_raw = _required(raw, "artifacts", "config")
        if not isinstance(source, dict) or not isinstance(scenario_raw, dict):
            raise ConfigError("source and scenario must be TOML tables")
        if not isinstance(fields_raw, dict) or not isinstance(artifacts_raw, list):
            raise ConfigError("fields must be a table and artifacts must be an array of tables")

        scenario = ScenarioDefaults(
            **{
                key: str(_required(scenario_raw, key, "scenario"))
                for key in ScenarioDefaults.__dataclass_fields__
            }
        )
        fields = FieldConfig(
            text=_tuple_of_strings(_required(fields_raw, "text", "fields"), "fields.text"),
            labels=_tuple_of_strings(_required(fields_raw, "labels", "fields"), "fields.labels"),
            record_id=_tuple_of_strings(
                _required(fields_raw, "record_id", "fields"), "fields.record_id"
            ),
        )

        artifacts: list[ArtifactConfig] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(artifacts_raw):
            context = f"artifacts[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{context} must be a table")
            artifact_id = str(_required(item, "id", context))
            artifact_format = str(_required(item, "format", context)).lower()
            if artifact_id in seen_ids:
                raise ConfigError(f"duplicate artifact id {artifact_id!r}")
            if artifact_format not in SUPPORTED_FORMATS:
                raise ConfigError(f"{context}: unsupported format {artifact_format!r}")
            digest = item.get("sha256") or None
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)
            ):
                raise ConfigError(f"{context}.sha256 must be a 64-character hexadecimal digest")
            artifacts.append(
                ArtifactConfig(
                    id=artifact_id,
                    split=str(_required(item, "split", context)),
                    format=artifact_format,
                    url=str(_required(item, "url", context)),
                    sha256=digest.lower() if digest else None,
                    records_path=str(item["records_path"]) if item.get("records_path") else None,
                    encoding=str(item.get("encoding", "utf-8")),
                )
            )
            seen_ids.add(artifact_id)
        if not artifacts:
            raise ConfigError("config must declare at least one artifact")

        return cls(
            path=config_path.resolve(),
            schema_version=schema_version,
            id=str(_required(source, "id", "source")),
            name=str(_required(source, "name", "source")),
            homepage=str(_required(source, "homepage", "source")),
            repository=str(_required(source, "repository", "source")),
            version=str(source["version"]) if source.get("version") is not None else None,
            revision=str(source["revision"]) if source.get("revision") is not None else None,
            license_manifest=str(_required(source, "license_manifest", "source")),
            importer=str(_required(source, "importer", "source")),
            scenario=scenario,
            fields=fields,
            artifacts=tuple(artifacts),
        )
