"""Offline JSON Schema loading and validation for PolicyBench contracts."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CHECKOUT_SCHEMA_DIRECTORY = _REPOSITORY_ROOT / "schemas"
_INSTALLED_SCHEMA_DIRECTORY = Path(sys.prefix) / "share" / "promptsec-dataset" / "schemas"

POLICY_CATALOG_SCHEMA = "policybench-policy-catalog-v0.1.schema.json"
SCENARIO_BLUEPRINT_SCHEMA = "policybench-scenario-blueprint-v0.1.schema.json"
GENERATION_RESPONSE_SCHEMA = "policybench-generation-response-v0.1.schema.json"
POLICYBENCH_RECORD_SCHEMA = "promptsec-policybench-record-v0.1.schema.json"

_SCHEMA_ALIASES = {
    "policy_catalog": POLICY_CATALOG_SCHEMA,
    "scenario_blueprint": SCENARIO_BLUEPRINT_SCHEMA,
    "generation_response": GENERATION_RESPONSE_SCHEMA,
    "policybench_record": POLICYBENCH_RECORD_SCHEMA,
}


class ContractValidationError(ValueError):
    """Raised when a value violates a PolicyBench JSON Schema contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        if not self.errors:
            raise ValueError("ContractValidationError requires at least one error")
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"PolicyBench contract validation failed:\n{details}")


def schema_path(schema: str | Path) -> Path:
    """Resolve a PolicyBench schema alias, filename, or explicit path."""

    candidate = Path(schema)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()

    filename = _SCHEMA_ALIASES.get(str(schema), str(schema))
    for directory in (_CHECKOUT_SCHEMA_DIRECTORY, _INSTALLED_SCHEMA_DIRECTORY):
        path = directory / filename
        if path.is_file():
            return path.resolve()
    return (_CHECKOUT_SCHEMA_DIRECTORY / filename).resolve()


@lru_cache(maxsize=32)
def _load_schema_path(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"PolicyBench schema not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON schema {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON schema must be an object: {path}")
    Draft202012Validator.check_schema(value)
    return value


def load_schema(schema: str | Path) -> dict[str, Any]:
    """Load and validate one schema document."""

    return _load_schema_path(schema_path(schema))


@lru_cache(maxsize=32)
def _build_validator(path: Path) -> Draft202012Validator:
    profile_path = path.resolve()
    profile = _load_schema_path(profile_path)
    registry = Registry()

    # Register every local schema by both canonical $id and file URI. This keeps
    # validation offline while allowing PolicyBench profiles to compose the
    # existing canonical dataset-record contract.
    for sibling in sorted(profile_path.parent.glob("*.json"), key=lambda item: item.name):
        try:
            document = _load_schema_path(sibling.resolve())
        except (ValueError, FileNotFoundError):
            if sibling.resolve() == profile_path:
                raise
            continue
        resource = Resource.from_contents(document)
        schema_id = document.get("$id")
        if isinstance(schema_id, str) and schema_id:
            registry = registry.with_resource(schema_id, resource)
        registry = registry.with_resource(sibling.resolve().as_uri(), resource)

    return Draft202012Validator(
        profile,
        registry=registry,
        format_checker=FormatChecker(),
    )


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


def _format_error(error: JsonSchemaValidationError) -> str:
    return f"{_json_path(error.absolute_path)}: {error.message}"


def _path_sort_key(parts: Sequence[Any]) -> tuple[str, ...]:
    return tuple(f"{type(part).__name__}:{part}" for part in parts)


def validate_instance(value: Any, schema: str | Path) -> list[str]:
    """Return deterministic, readable validation errors for one instance."""

    validator = _build_validator(schema_path(schema))
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (_path_sort_key(error.absolute_path), error.message),
    )
    return list(dict.fromkeys(_format_error(error) for error in errors))


def require_valid_instance(value: Any, schema: str | Path) -> None:
    """Raise :class:`ContractValidationError` unless ``value`` is valid."""

    errors = validate_instance(value, schema)
    if errors:
        raise ContractValidationError(errors)


def validate_policy_catalog_schema(value: Any) -> list[str]:
    return validate_instance(value, POLICY_CATALOG_SCHEMA)


def validate_scenario_blueprint_schema(value: Any) -> list[str]:
    return validate_instance(value, SCENARIO_BLUEPRINT_SCHEMA)


def validate_generation_response_schema(value: Any) -> list[str]:
    return validate_instance(value, GENERATION_RESPONSE_SCHEMA)


def validate_policybench_record_schema(value: Any) -> list[str]:
    return validate_instance(value, POLICYBENCH_RECORD_SCHEMA)


def require_valid_policybench_record(value: Mapping[str, Any]) -> None:
    require_valid_instance(value, POLICYBENCH_RECORD_SCHEMA)


__all__ = [
    "ContractValidationError",
    "GENERATION_RESPONSE_SCHEMA",
    "POLICYBENCH_RECORD_SCHEMA",
    "POLICY_CATALOG_SCHEMA",
    "SCENARIO_BLUEPRINT_SCHEMA",
    "load_schema",
    "require_valid_instance",
    "require_valid_policybench_record",
    "schema_path",
    "validate_generation_response_schema",
    "validate_instance",
    "validate_policy_catalog_schema",
    "validate_policybench_record_schema",
    "validate_scenario_blueprint_schema",
]
