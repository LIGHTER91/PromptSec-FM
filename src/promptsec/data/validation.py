"""Validation for canonical PromptSec-FM dataset records.

The frozen annotation schema remains the normative source for labels.  The
dataset-record profile adds reproducibility metadata, while this module adds
the semantic invariants that JSON Schema cannot express conveniently.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource

from promptsec.data.hashing import sha256_json, sha256_text

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CHECKOUT_SCHEMA_PATH = _REPOSITORY_ROOT / "schemas" / "promptsec-dataset-record-v0.1.schema.json"
_INSTALLED_SCHEMA_PATH = (
    Path(sys.prefix)
    / "share"
    / "promptsec-dataset"
    / "schemas"
    / "promptsec-dataset-record-v0.1.schema.json"
)
_ANNOTATION_SCHEMA_NAME = "promptsec-annotation-v1.schema.json"
_DATASET_SCHEMA_NAME = "promptsec-dataset-record-v0.1.schema.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_VERDICT_AXES = (
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "authority_status",
)
_AXIS_VALUES = {
    "instruction_presence": {
        "NO_INSTRUCTION",
        "INSTRUCTION_PRESENT",
        "UNDETERMINED",
    },
    "instruction_presentation": {
        "OPERATIVE",
        "QUOTED_OR_REPORTED",
        "HYPOTHETICAL",
        "UNKNOWN",
        "NOT_APPLICABLE",
    },
    "instruction_addressee": {
        "MODEL_OR_AGENT",
        "HUMAN",
        "EXTERNAL_SYSTEM",
        "UNKNOWN",
        "NOT_APPLICABLE",
    },
    "authority_status": {
        "WITHIN_AUTHORITY",
        "OUTSIDE_AUTHORITY",
        "SPOOFED",
        "UNKNOWN",
        "NOT_APPLICABLE",
    },
    "annotation_status": {
        "CONFIRMED",
        "INSUFFICIENT_CONTEXT",
        "DISAGREEMENT",
        "EXCLUDED",
    },
}


class ValidationError(ValueError):
    """Raised when a dataset record violates one or more invariants."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        if not self.errors:
            raise ValueError("ValidationError requires at least one error")
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"PromptSec record validation failed:\n{details}")


def derive_prompt_injection_verdict(annotations: Mapping[str, Any]) -> str:
    """Derive the normative v1.0 prompt-injection verdict.

    The four instruction/authority axes and ``annotation_status`` must be
    present.  Schema validation normally guarantees this; requiring them here
    prevents a partial annotation from silently becoming ``NOT_DETECTED``.
    """

    required = {*_VERDICT_AXES, "annotation_status"}
    missing = sorted(required.difference(annotations))
    if missing:
        raise ValueError(
            "cannot derive prompt_injection_verdict; missing annotation fields: "
            + ", ".join(missing)
        )

    for field in sorted(required):
        value = annotations[field]
        if not isinstance(value, str) or value not in _AXIS_VALUES[field]:
            raise ValueError(f"cannot derive prompt_injection_verdict; invalid {field}: {value!r}")

    if (
        annotations["instruction_presence"] == "INSTRUCTION_PRESENT"
        and annotations["instruction_presentation"] == "OPERATIVE"
        and annotations["instruction_addressee"] == "MODEL_OR_AGENT"
        and annotations["authority_status"] in {"OUTSIDE_AUTHORITY", "SPOOFED"}
    ):
        return "DETECTED"

    if annotations["annotation_status"] in {
        "INSUFFICIENT_CONTEXT",
        "DISAGREEMENT",
    }:
        return "UNCERTAIN"

    unresolved = (
        annotations["instruction_presence"] == "UNDETERMINED"
        or annotations["instruction_presentation"] == "UNKNOWN"
        or annotations["instruction_addressee"] == "UNKNOWN"
        or annotations["authority_status"] == "UNKNOWN"
    )
    return "UNCERTAIN" if unresolved else "NOT_DETECTED"


def validate_record(
    record: Any,
    schema_path: str | Path | None = None,
) -> list[str]:
    """Return all readable schema and semantic errors for ``record``.

    An empty list means that the record is valid.  ``schema_path`` can select a
    compatible dataset-record profile; its sibling frozen annotation schema is
    registered locally, so validation never resolves ``$ref`` over the network.
    """

    validator = _build_validator(Path(schema_path) if schema_path else _default_schema_path())
    schema_errors = sorted(
        validator.iter_errors(record),
        key=lambda error: (_path_sort_key(error.absolute_path), error.message),
    )
    errors = [_format_schema_error(error) for error in schema_errors]

    if isinstance(record, Mapping):
        errors.extend(_semantic_errors(record))

    # Conditional/allOf failures can occasionally surface the same leaf error
    # more than once.  Preserve deterministic order while removing duplicates.
    return list(dict.fromkeys(errors))


def require_valid_record(
    record: Any,
    schema_path: str | Path | None = None,
) -> None:
    """Raise :class:`ValidationError` unless ``record`` is fully valid."""

    errors = validate_record(record, schema_path=schema_path)
    if errors:
        raise ValidationError(errors)


@lru_cache(maxsize=8)
def _build_validator(schema_path: Path) -> Draft202012Validator:
    profile_path = schema_path.resolve()
    annotation_path = profile_path.with_name(_ANNOTATION_SCHEMA_NAME)
    if not annotation_path.is_file():
        fallback = _default_schema_path().with_name(_ANNOTATION_SCHEMA_NAME)
        if fallback.is_file():
            annotation_path = fallback

    profile = _load_json_object(profile_path)
    annotation = _load_json_object(annotation_path)
    Draft202012Validator.check_schema(profile)
    Draft202012Validator.check_schema(annotation)

    annotation_resource = Resource.from_contents(annotation)
    annotation_id = annotation.get("$id")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError(f"annotation schema has no $id: {annotation_path}")

    registry = Registry().with_resource(annotation_id, annotation_resource)
    registry = registry.with_resource(annotation_path.as_uri(), annotation_resource)
    dataset_path = profile_path.with_name(_DATASET_SCHEMA_NAME)
    if profile_path != dataset_path and dataset_path.is_file():
        dataset = _load_json_object(dataset_path)
        Draft202012Validator.check_schema(dataset)
        dataset_id = dataset.get("$id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"dataset schema has no $id: {dataset_path}")
        dataset_resource = Resource.from_contents(dataset)
        registry = registry.with_resource(dataset_id, dataset_resource)
        registry = registry.with_resource(dataset_path.as_uri(), dataset_resource)
    return Draft202012Validator(
        profile,
        registry=registry,
        format_checker=FormatChecker(),
    )


def _default_schema_path() -> Path:
    for candidate in (_CHECKOUT_SCHEMA_PATH, _INSTALLED_SCHEMA_PATH):
        if candidate.is_file():
            return candidate
    # Keep the checkout location in the eventual error because it is the most
    # actionable path for contributors and editable installations.
    return _CHECKOUT_SCHEMA_PATH


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"PromptSec schema not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON schema {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"JSON schema must be an object: {path}")
    return value


def _format_schema_error(error: JsonSchemaValidationError) -> str:
    return f"{_json_path(error.absolute_path)}: {error.message}"


def _json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part.isidentifier():
            path += f".{part}"
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def _path_sort_key(parts: Sequence[Any]) -> tuple[str, ...]:
    return tuple(f"{type(part).__name__}:{part}" for part in parts)


def _semantic_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    content = record.get("content")
    annotations = record.get("annotations")
    derived = record.get("derived")

    text = content.get("text") if isinstance(content, Mapping) else None
    if isinstance(annotations, Mapping):
        errors.extend(_span_errors(annotations.get("spans"), text))
        errors.extend(_fallback_errors(annotations))
        errors.extend(_verdict_errors(annotations, derived))

    if isinstance(text, str):
        errors.extend(_checksum_errors(record, text))
    return errors


def _span_errors(spans: Any, text: Any) -> list[str]:
    if not isinstance(spans, list) or not isinstance(text, str):
        return []

    errors: list[str] = []
    text_length = len(text)
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            continue
        start = span.get("start")
        end = span.get("end")
        if not _is_integer(start) or not _is_integer(end):
            continue
        if not 0 <= start < end <= text_length:
            errors.append(
                f"$.annotations.spans[{index}]: expected "
                f"0 <= start < end <= len(content.text) ({text_length}); "
                f"got start={start}, end={end}"
            )
    return errors


def _fallback_errors(annotations: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    families = annotations.get("attack_families")
    if isinstance(families, list) and "OTHER_PROMPT_ATTACK" in families and len(families) != 1:
        errors.append(
            "$.annotations.attack_families: OTHER_PROMPT_ATTACK is a fallback "
            "and must be the only attack family"
        )

    objectives = annotations.get("attack_objectives")
    if isinstance(objectives, list) and "OTHER" in objectives and len(objectives) != 1:
        errors.append(
            "$.annotations.attack_objectives: OTHER is a fallback and must be "
            "the only attack objective"
        )
    return errors


def _verdict_errors(annotations: Mapping[str, Any], derived: Any) -> list[str]:
    if not isinstance(derived, Mapping):
        return []
    actual = derived.get("prompt_injection_verdict")
    if not isinstance(actual, str):
        return []

    required = {*_VERDICT_AXES, "annotation_status"}
    if not required.issubset(annotations):
        return []
    if any(
        not isinstance(annotations[field], str) or annotations[field] not in _AXIS_VALUES[field]
        for field in required
    ):
        return []

    expected = derive_prompt_injection_verdict(annotations)
    if actual == expected:
        return []
    return [
        f"$.derived.prompt_injection_verdict: expected derived verdict {expected!r}, got {actual!r}"
    ]


def _checksum_errors(record: Mapping[str, Any], canonical_text: str) -> list[str]:
    provenance = _nested_mapping(record, "metadata", "dataset_provenance")
    if provenance is None:
        return []
    checksums = provenance.get("checksums")
    if not isinstance(checksums, Mapping):
        return []

    errors: list[str] = []
    canonical_checksum = checksums.get("canonical_text_sha256")
    expected_canonical = sha256_text(canonical_text)
    if (
        isinstance(canonical_checksum, str)
        and _SHA256_RE.fullmatch(canonical_checksum)
        and canonical_checksum != expected_canonical
    ):
        errors.append(
            "$.metadata.dataset_provenance.checksums.canonical_text_sha256: "
            f"does not match content.text (expected {expected_canonical})"
        )

    source_record = provenance.get("source_record")
    if isinstance(source_record, Mapping):
        original_fields = source_record.get("original_fields")
        raw_record_checksum = source_record.get("raw_record_sha256")
        if (
            isinstance(original_fields, Mapping)
            and isinstance(raw_record_checksum, str)
            and _SHA256_RE.fullmatch(raw_record_checksum)
        ):
            try:
                expected_raw_record = sha256_json(original_fields)
            except (TypeError, ValueError, UnicodeEncodeError):
                errors.append(
                    "$.metadata.dataset_provenance.source_record.original_fields: "
                    "cannot be serialized as canonical JSON"
                )
            else:
                if raw_record_checksum != expected_raw_record:
                    errors.append(
                        "$.metadata.dataset_provenance.source_record.raw_record_sha256: "
                        "does not match original_fields "
                        f"(expected {expected_raw_record})"
                    )

    source_text = _mapped_source_text(provenance)
    source_checksum = checksums.get("source_text_sha256")
    if (
        source_text is not None
        and isinstance(source_checksum, str)
        and _SHA256_RE.fullmatch(source_checksum)
    ):
        expected_source = sha256_text(source_text)
        if source_checksum != expected_source:
            errors.append(
                "$.metadata.dataset_provenance.checksums.source_text_sha256: "
                f"does not match mapped source text (expected {expected_source})"
            )
    return errors


def _mapped_source_text(provenance: Mapping[str, Any]) -> str | None:
    source_record = provenance.get("source_record")
    mapping = provenance.get("mapping")
    if not isinstance(source_record, Mapping) or not isinstance(mapping, Mapping):
        return None
    original_fields = source_record.get("original_fields")
    field_mappings = mapping.get("field_mappings")
    if not isinstance(original_fields, Mapping) or not isinstance(field_mappings, list):
        return None

    candidates: list[str] = []
    for field_mapping in field_mappings:
        if not isinstance(field_mapping, Mapping):
            continue
        target = field_mapping.get("target")
        source = field_mapping.get("source")
        if target not in {"content.text", "$.content.text", "/content/text"}:
            continue
        if not isinstance(source, str):
            continue
        value = _resolve_source_path(original_fields, source)
        if isinstance(value, str):
            candidates.append(value)

    return candidates[0] if len(candidates) == 1 else None


def _resolve_source_path(original_fields: Mapping[str, Any], path: str) -> Any:
    # Exact source keys take precedence because dataset column names may contain
    # dots.  JSON Pointer and dotted paths are supported for nested records.
    if path in original_fields:
        return original_fields[path]

    if path.startswith("/"):
        parts = [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]
    else:
        normalized = path[2:] if path.startswith("$.") else path
        parts = normalized.split(".") if normalized else []

    value: Any = original_fields
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "ValidationError",
    "derive_prompt_injection_verdict",
    "require_valid_record",
    "validate_record",
]
