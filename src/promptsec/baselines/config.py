"""Configuration loading and validation for the fixed CPU benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

TARGETS = (
    "prompt_injection_verdict",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "instruction_presentation",
)
ABLATIONS = ("M1", "M2", "M3", "M4")
MODEL_FAMILIES = (
    "WORD_TFIDF_LOGREG",
    "CHAR_TFIDF_LOGREG",
    "WORD_CHAR_TFIDF_LINEAR_SVM",
)
EVALUATION_SPLITS = (
    "validation",
    "test_policy_family_ood",
    "test_domain_ood",
    "test_language_ood",
    "test_counterfactual",
)


class BaselineConfigError(ValueError):
    """Raised when the benchmark configuration is not the frozen v0.1 matrix."""


def _require_names(config: Mapping[str, Any], key: str, expected: Sequence[str]) -> None:
    actual = config.get(key)
    if not isinstance(actual, list) or tuple(actual) != tuple(expected):
        raise BaselineConfigError(f"{key} must be exactly {list(expected)!r}")


def load_baseline_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and enforce the deliberately small, fixed experiment matrix."""

    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise BaselineConfigError(f"cannot load baseline config {source}: {error}") from error
    if not isinstance(value, dict):
        raise BaselineConfigError("baseline config must be an object")
    if value.get("schema_version") != "0.1":
        raise BaselineConfigError("schema_version must be '0.1'")
    _require_names(value, "targets", TARGETS)
    _require_names(value, "ablations", ABLATIONS)
    _require_names(value, "model_families", MODEL_FAMILIES)
    _require_names(value, "evaluation_splits", EVALUATION_SPLITS)
    if value.get("seed") != 20260718:
        raise BaselineConfigError("the v0.1 default seed must be 20260718")
    if not isinstance(value.get("models"), dict):
        raise BaselineConfigError("models must be an object")
    missing = set(MODEL_FAMILIES).difference(value["models"])
    if missing:
        raise BaselineConfigError(f"models are missing configurations: {sorted(missing)}")
    return value


def select_names(
    requested: Sequence[str] | None,
    allowed: Sequence[str],
    kind: str,
) -> tuple[str, ...]:
    """Validate CLI filters while preserving the canonical matrix order."""

    if not requested:
        return tuple(allowed)
    unknown = set(requested).difference(allowed)
    if unknown:
        raise BaselineConfigError(f"unknown {kind}: {sorted(unknown)}")
    selected = set(requested)
    return tuple(name for name in allowed if name in selected)
