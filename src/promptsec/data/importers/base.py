"""Base importer contract and canonical provenance construction."""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from promptsec.data.config import ArtifactConfig, SourceConfig
from promptsec.data.hashing import sha256_json, sha256_text
from promptsec.data.readers import read_records
from promptsec.data.validation import derive_prompt_injection_verdict


class ImporterError(ValueError):
    """Raised when an importer cannot convert a source record safely."""


@dataclass(frozen=True, slots=True)
class RawRecord:
    artifact: ArtifactConfig
    raw_artifact: str
    split: str
    index: int
    payload: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def portable_path(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def uncertain_annotations() -> dict[str, Any]:
    """Return a conservative v1 annotation when source context is insufficient."""

    return {
        "instruction_presence": "UNDETERMINED",
        "instruction_presentation": "UNKNOWN",
        "instruction_addressee": "UNKNOWN",
        "user_goal_alignment": "UNDETERMINED",
        "protected_policy_alignment": "UNDETERMINED",
        "authority_status": "UNKNOWN",
        "attack_families": [],
        "attack_objectives": [],
        "spans": [],
        "annotation_status": "INSUFFICIENT_CONTEXT",
        "annotator_confidence": 0.0,
    }


class BaseImporter(ABC):
    """Convert immutable source objects into canonical PromptSec records.

    Subclasses interpret source fields, but this class owns stable identifiers,
    source-record preservation, corpus provenance, and scenario-provenance defaults.
    """

    importer_version = "0.1.0"

    def __init__(self, config: SourceConfig, imported_at: str | None = None) -> None:
        self.config = config
        self.imported_at = imported_at or utc_now()

    def iter_raw_records(self, inputs: Mapping[str, str | Path]) -> Iterator[RawRecord]:
        expected = {artifact.id for artifact in self.config.artifacts}
        unexpected = set(inputs) - expected
        if unexpected:
            raise ImporterError(f"unknown input artifact(s): {sorted(unexpected)}")
        missing = expected - set(inputs)
        if missing:
            raise ImporterError(f"missing input artifact(s): {sorted(missing)}")

        for artifact in self.config.artifacts:
            source_path = Path(inputs[artifact.id])
            for index, payload in enumerate(read_records(source_path, artifact)):
                yield RawRecord(
                    artifact=artifact,
                    raw_artifact=portable_path(source_path),
                    split=artifact.split,
                    index=index,
                    payload=payload,
                )

    def records(self, inputs: Mapping[str, str | Path]) -> Iterator[dict[str, Any]]:
        for raw in self.iter_raw_records(inputs):
            yield from self.transform(raw)

    @abstractmethod
    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        """Map one raw source object to zero or more canonical records."""

    def first_present(self, payload: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, Any]:
        for field in fields:
            if field in payload and payload[field] is not None:
                return field, payload[field]
        raise ImporterError(f"none of the expected fields is present: {list(fields)}")

    def source_record_id(self, raw: RawRecord) -> str:
        for field in self.config.fields.record_id:
            value = raw.payload.get(field)
            if value is not None and str(value):
                return str(value)
        return f"{raw.artifact.id}:{raw.index}"

    def canonical_id(self, raw: RawRecord) -> str:
        source_id = self.source_record_id(raw)
        identity = {
            "dataset": self.config.id,
            "revision": self.config.revision,
            "version": self.config.version,
            "artifact": raw.artifact.id,
            "split": raw.split,
            "source_record_id": source_id,
        }
        slug = re.sub(r"[^a-z0-9]+", "_", self.config.id.lower()).strip("_") or "source"
        return f"psfm_{slug}_{sha256_json(identity)[:24]}"

    def build_record(
        self,
        raw: RawRecord,
        *,
        text: str,
        source_text: str,
        annotations: Mapping[str, Any],
        original_labels: Mapping[str, Any],
        mapping_status: str,
        field_mappings: Sequence[Mapping[str, str]],
        unmapped_labels: Sequence[str] = (),
        review_reasons: Sequence[str] = (),
        transformations: Sequence[str] = (),
        context: Mapping[str, Any] | None = None,
        content_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not isinstance(source_text, str):
            raise ImporterError("source and canonical text must be strings")
        annotation_object = dict(annotations)
        scenario = self.config.scenario
        content = {
            "text": text,
            "language": scenario.language,
            "delivery_mode": scenario.delivery_mode,
            "source_role": scenario.source_role,
            "content_origin": scenario.content_origin,
            "ingestion_path": scenario.ingestion_path,
            "modality": scenario.modality,
            "source_integrity": scenario.source_integrity,
        }
        content.update(dict(content_overrides or {}))
        record_context = {
            "protected_policy": None,
            "user_goal": None,
            "available_capabilities": [],
        }
        record_context.update(dict(context or {}))

        return {
            "id": self.canonical_id(raw),
            "taxonomy_version": "1.0",
            "context": record_context,
            "content": content,
            "annotations": annotation_object,
            "derived": {
                "prompt_injection_verdict": derive_prompt_injection_verdict(annotation_object)
            },
            "metadata": {
                "record_schema_version": "0.1",
                "dataset_provenance": {
                    "source_dataset": {
                        "id": self.config.id,
                        "name": self.config.name,
                        "version": self.config.version,
                        "revision": self.config.revision,
                        "url": self.config.homepage,
                        "license_manifest": self.config.license_manifest,
                    },
                    "source_record": {
                        "id": self.source_record_id(raw),
                        "split": raw.split,
                        "index": raw.index,
                        "raw_artifact": raw.raw_artifact,
                        "raw_record_sha256": sha256_json(raw.payload),
                        "original_fields": raw.payload,
                        "original_labels": dict(original_labels),
                    },
                    "import": {
                        "importer": type(self).__name__,
                        "importer_version": self.importer_version,
                        "config": portable_path(self.config.path),
                        "config_sha256": self.config.sha256,
                        "imported_at": self.imported_at,
                        "transformations": list(transformations),
                    },
                    "mapping": {
                        "ruleset": "taxonomy_migration_v1",
                        "status": mapping_status,
                        "field_mappings": [dict(item) for item in field_mappings],
                        "unmapped_labels": list(unmapped_labels),
                        "review_reasons": list(review_reasons),
                    },
                    "checksums": {
                        "source_text_sha256": sha256_text(source_text),
                        "canonical_text_sha256": sha256_text(text),
                    },
                },
            },
        }


def load_importer(spec: str, config: SourceConfig, imported_at: str | None = None) -> BaseImporter:
    """Load a configured importer from ``module:ClassName``."""

    if ":" not in spec:
        raise ImporterError("importer must use the form 'module:ClassName'")
    module_name, class_name = spec.split(":", 1)
    try:
        importer_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImporterError(f"cannot load importer {spec!r}: {exc}") from exc
    if not isinstance(importer_class, type) or not issubclass(importer_class, BaseImporter):
        raise ImporterError(f"configured importer {spec!r} is not a BaseImporter subclass")
    return importer_class(config, imported_at=imported_at)
