"""Strict, reproducible configuration for PromptSec-PolicyBench v0.1."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

DOMAIN_ORDER = (
    "banking",
    "email",
    "calendar",
    "file_management",
    "web_and_purchases",
    "persistent_memory",
)
LANGUAGE_ORDER = ("en", "fr")
CATEGORY_ORDER = (
    "NO_INSTRUCTION",
    "ALIGNED_AND_COMPLIANT",
    "ALIGNED_BUT_POLICY_CONFLICTING",
    "MISALIGNED_AND_POLICY_CONFLICTING",
    "MISALIGNED_NOT_POLICY_CONFLICTING",
    "QUOTED_OR_REPORTED",
    "HYPOTHETICAL",
    "SPOOFED_AUTHORITY",
    "INSUFFICIENT_CONTEXT",
    "HARD_NEGATIVE_SPECIAL_CASES",
)
COUNTERFACTUAL_TYPE_ORDER = (
    "POLICY_CHANGE",
    "USER_GOAL_CHANGE",
    "SOURCE_ROLE_CHANGE",
    "AUTHORITY_DELEGATION_CHANGE",
    "CAPABILITY_CHANGE",
    "PRESENTATION_CHANGE",
)
SPLIT_GROUP_FIELDS = (
    "policy_family",
    "scenario_template_family",
    "attack_template_family",
    "counterfactual_group_id",
    "base_generation_family",
    "semantic_duplicate_cluster",
)

_ROOT_KEYS = {
    "schema_version",
    "release_id",
    "taxonomy_version",
    "record_schema_version",
    "seed",
    "generated_at",
    "target_records",
    "languages",
    "domains",
    "category_quotas",
    "counterfactual_quotas",
    "generation",
    "quality",
    "split_strategy",
    "paths",
    "review_sample_size",
}
_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-(.*?))?\}")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class PolicyBenchConfigError(ValueError):
    """Raised when a PolicyBench configuration is incomplete or unsafe."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects mappings with repeated keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise PolicyBenchConfigError("YAML mapping keys must be scalar values") from error
        if duplicate:
            raise PolicyBenchConfigError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PolicyBenchConfigError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected.difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise PolicyBenchConfigError(f"{context} keys are invalid: {'; '.join(details)}")


def _string(value: Any, context: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise PolicyBenchConfigError(f"{context} must be a non-empty string{suffix}")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyBenchConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    context: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyBenchConfigError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyBenchConfigError(f"{context} must be a finite number")
    if result < minimum or (strictly_positive and result <= minimum):
        operator = ">" if strictly_positive else ">="
        raise PolicyBenchConfigError(f"{context} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise PolicyBenchConfigError(f"{context} must be <= {maximum}")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyBenchConfigError(f"{context} must be a boolean")
    return value


def _resolve_environment(value: Any, environ: Mapping[str, str], context: str = "config") -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_environment(item, environ, f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_environment(item, environ, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str) or "${" not in value:
        return value

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environ:
            return environ[name]
        if default is not None:
            return default
        raise PolicyBenchConfigError(f"{context}: environment variable {name!r} is not set")

    resolved = _ENV_PATTERN.sub(substitute, value)
    if "${" in resolved:
        raise PolicyBenchConfigError(f"{context}: malformed environment placeholder")
    return resolved


def _quota_mapping(
    value: Any,
    names: Sequence[str],
    context: str,
    *,
    integer_values: bool = False,
    require_sum: float | None = None,
) -> dict[str, int] | dict[str, float]:
    table = _mapping(value, context)
    _exact_keys(table, set(names), context)
    if integer_values:
        return {name: _integer(table[name], f"{context}.{name}", minimum=1) for name in names}
    result = {
        name: _number(table[name], f"{context}.{name}", strictly_positive=True) for name in names
    }
    if require_sum is not None and not math.isclose(
        sum(result.values()), require_sum, abs_tol=1e-9
    ):
        raise PolicyBenchConfigError(f"{context} values must sum to {require_sum}")
    return result


def apportion_quota(
    total: int,
    weights: Mapping[str, int | float],
    *,
    order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Use deterministic largest-remainder apportionment with an explicit tie order."""

    total = _integer(total, "total", minimum=0)
    if not isinstance(weights, Mapping) or not weights:
        raise PolicyBenchConfigError("weights must be a non-empty mapping")
    names = tuple(order) if order is not None else tuple(sorted(weights))
    if set(names) != set(weights) or len(names) != len(weights):
        raise PolicyBenchConfigError("order must contain each weight key exactly once")
    decimals: dict[str, Decimal] = {}
    for name in names:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyBenchConfigError(f"weight {name!r} must be numeric")
        decimal = Decimal(str(value))
        if not decimal.is_finite() or decimal < 0:
            raise PolicyBenchConfigError(f"weight {name!r} must be finite and non-negative")
        decimals[name] = decimal
    denominator = sum(decimals.values(), Decimal(0))
    if denominator <= 0:
        raise PolicyBenchConfigError("at least one weight must be positive")

    exact = {name: Decimal(total) * decimals[name] / denominator for name in names}
    allocated = {name: int(exact[name].to_integral_value(rounding=ROUND_FLOOR)) for name in names}
    remainder = total - sum(allocated.values())
    rank = {name: index for index, name in enumerate(names)}
    recipients = sorted(
        names,
        key=lambda name: (-(exact[name] - allocated[name]), rank[name]),
    )
    for name in recipients[:remainder]:
        allocated[name] += 1
    return allocated


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    provider: str
    model: str
    model_revision: str | None
    base_url: str
    api_key_env: str
    authentication: str
    response_mode: str
    temperature: float
    max_retries: int
    concurrency: int
    timeout_seconds: float
    max_response_bytes: int
    records_per_batch: int
    reasoning_effort: str | None
    ephemeral: bool
    output_schema: str
    max_prompt_bytes: int
    codex_executable: str

    @classmethod
    def from_mapping(cls, value: Any) -> GenerationConfig:
        table = dict(_mapping(value, "generation"))
        # Preserve the original v0.1 configuration contract while allowing new,
        # explicit local-provider controls in dedicated configurations.
        table.setdefault("base_url", "https://api.openai.com/v1")
        table.setdefault("api_key_env", "PROMPTSEC_GENERATION_API_KEY")
        table.setdefault("authentication", "required")
        table.setdefault("response_mode", "strict_json_schema")
        table.setdefault("records_per_batch", 1)
        table.setdefault("reasoning_effort", None)
        table.setdefault("ephemeral", False)
        table.setdefault("output_schema", "strict")
        table.setdefault("max_prompt_bytes", 1_048_576)
        table.setdefault("codex_executable", "codex")
        expected = {
            "provider",
            "model",
            "model_revision",
            "base_url",
            "api_key_env",
            "authentication",
            "response_mode",
            "temperature",
            "max_retries",
            "concurrency",
            "timeout_seconds",
            "max_response_bytes",
            "records_per_batch",
            "reasoning_effort",
            "ephemeral",
            "output_schema",
            "max_prompt_bytes",
            "codex_executable",
        }
        _exact_keys(table, expected, "generation")
        api_key_env = _string(table["api_key_env"], "generation.api_key_env")
        assert isinstance(api_key_env, str)
        if not _ENV_NAME.fullmatch(api_key_env):
            raise PolicyBenchConfigError(
                "generation.api_key_env must name an uppercase environment variable"
            )
        provider = _string(table["provider"], "generation.provider")
        model = _string(table["model"], "generation.model")
        base_url = _string(table["base_url"], "generation.base_url")
        authentication = _string(table["authentication"], "generation.authentication")
        if authentication not in {"required", "optional_for_loopback"}:
            raise PolicyBenchConfigError(
                "generation.authentication must be 'required' or 'optional_for_loopback'"
            )
        response_mode = _string(table["response_mode"], "generation.response_mode")
        if response_mode not in {
            "strict_json_schema",
            "json_object",
            "prompt_constrained_json",
        }:
            raise PolicyBenchConfigError(
                "generation.response_mode must be 'strict_json_schema', 'json_object', "
                "or 'prompt_constrained_json'"
            )
        records_per_batch = _integer(
            table["records_per_batch"], "generation.records_per_batch", minimum=1
        )
        if records_per_batch not in {1, 2, 5, 10, 20}:
            raise PolicyBenchConfigError(
                "generation.records_per_batch must be one of 1, 2, 5, 10, or 20"
            )
        reasoning_effort = _string(
            table["reasoning_effort"], "generation.reasoning_effort", nullable=True
        )
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise PolicyBenchConfigError(
                "generation.reasoning_effort must be low, medium, high, xhigh, or null"
            )
        output_schema = _string(table["output_schema"], "generation.output_schema")
        if output_schema != "strict":
            raise PolicyBenchConfigError("generation.output_schema must equal 'strict'")
        codex_executable = _string(table["codex_executable"], "generation.codex_executable")
        assert isinstance(codex_executable, str)
        if "\x00" in codex_executable:
            raise PolicyBenchConfigError("generation.codex_executable contains a null byte")
        assert isinstance(provider, str) and isinstance(model, str) and isinstance(base_url, str)
        return cls(
            provider=provider,
            model=model,
            model_revision=_string(
                table["model_revision"], "generation.model_revision", nullable=True
            ),
            base_url=base_url,
            api_key_env=api_key_env,
            authentication=authentication,
            response_mode=response_mode,
            temperature=_number(table["temperature"], "generation.temperature", maximum=2.0),
            max_retries=_integer(table["max_retries"], "generation.max_retries"),
            concurrency=_integer(table["concurrency"], "generation.concurrency", minimum=1),
            timeout_seconds=_number(
                table["timeout_seconds"],
                "generation.timeout_seconds",
                strictly_positive=True,
            ),
            max_response_bytes=_integer(
                table["max_response_bytes"],
                "generation.max_response_bytes",
                minimum=1,
            ),
            records_per_batch=records_per_batch,
            reasoning_effort=reasoning_effort,
            ephemeral=_boolean(table["ephemeral"], "generation.ephemeral"),
            output_schema=output_schema,
            max_prompt_bytes=_integer(
                table["max_prompt_bytes"], "generation.max_prompt_bytes", minimum=1
            ),
            codex_executable=codex_executable,
        )


@dataclass(frozen=True, slots=True)
class QualityConfig:
    exact_duplicate_rejection: bool
    normalized_duplicate_rejection: bool
    semantic_duplicate_threshold: float
    require_counterfactual_validation: bool
    require_span_validation: bool
    maximum_candidate_characters: int
    maximum_failed_attempt_log_characters: int

    @classmethod
    def from_mapping(cls, value: Any) -> QualityConfig:
        table = _mapping(value, "quality")
        expected = {
            "exact_duplicate_rejection",
            "normalized_duplicate_rejection",
            "semantic_duplicate_threshold",
            "require_counterfactual_validation",
            "require_span_validation",
            "maximum_candidate_characters",
            "maximum_failed_attempt_log_characters",
        }
        _exact_keys(table, expected, "quality")
        return cls(
            exact_duplicate_rejection=_boolean(
                table["exact_duplicate_rejection"], "quality.exact_duplicate_rejection"
            ),
            normalized_duplicate_rejection=_boolean(
                table["normalized_duplicate_rejection"],
                "quality.normalized_duplicate_rejection",
            ),
            semantic_duplicate_threshold=_number(
                table["semantic_duplicate_threshold"],
                "quality.semantic_duplicate_threshold",
                maximum=1.0,
            ),
            require_counterfactual_validation=_boolean(
                table["require_counterfactual_validation"],
                "quality.require_counterfactual_validation",
            ),
            require_span_validation=_boolean(
                table["require_span_validation"], "quality.require_span_validation"
            ),
            maximum_candidate_characters=_integer(
                table["maximum_candidate_characters"],
                "quality.maximum_candidate_characters",
                minimum=1,
            ),
            maximum_failed_attempt_log_characters=_integer(
                table["maximum_failed_attempt_log_characters"],
                "quality.maximum_failed_attempt_log_characters",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class SplitStrategyConfig:
    seed: int
    train_ratio: float
    validation_ratio: float
    policy_family_ood_ratio: float
    domain_ood_ratio: float
    language_ood_ratio: float
    counterfactual_ratio: float
    held_out_domain: str
    held_out_language: str
    group_fields: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> SplitStrategyConfig:
        table = _mapping(value, "split_strategy")
        expected = {
            "seed",
            "train_ratio",
            "validation_ratio",
            "policy_family_ood_ratio",
            "domain_ood_ratio",
            "language_ood_ratio",
            "counterfactual_ratio",
            "held_out_domain",
            "held_out_language",
            "group_fields",
        }
        _exact_keys(table, expected, "split_strategy")
        ratios = {
            name: _number(table[name], f"split_strategy.{name}", maximum=1.0)
            for name in (
                "train_ratio",
                "validation_ratio",
                "policy_family_ood_ratio",
                "domain_ood_ratio",
                "language_ood_ratio",
                "counterfactual_ratio",
            )
        }
        if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
            raise PolicyBenchConfigError("split_strategy ratios must sum to 1.0")
        group_fields = table["group_fields"]
        if not isinstance(group_fields, list) or not all(
            isinstance(item, str) and item for item in group_fields
        ):
            raise PolicyBenchConfigError("split_strategy.group_fields must be a string array")
        if tuple(group_fields) != SPLIT_GROUP_FIELDS:
            raise PolicyBenchConfigError(
                "split_strategy.group_fields must list every required leakage group in order"
            )
        held_out_domain = _string(table["held_out_domain"], "split_strategy.held_out_domain")
        held_out_language = _string(table["held_out_language"], "split_strategy.held_out_language")
        assert isinstance(held_out_domain, str) and isinstance(held_out_language, str)
        if held_out_domain not in DOMAIN_ORDER:
            raise PolicyBenchConfigError("split_strategy.held_out_domain is unsupported")
        if held_out_language not in LANGUAGE_ORDER:
            raise PolicyBenchConfigError("split_strategy.held_out_language is unsupported")
        return cls(
            seed=_integer(table["seed"], "split_strategy.seed"),
            held_out_domain=held_out_domain,
            held_out_language=held_out_language,
            group_fields=tuple(group_fields),
            **ratios,
        )


@dataclass(frozen=True, slots=True)
class OutputPaths:
    policies: str
    prompts: str
    raw_outputs: str
    accepted_artifacts: str
    output: str
    reports: str
    review: str

    @classmethod
    def from_mapping(cls, value: Any) -> OutputPaths:
        table = _mapping(value, "paths")
        expected = {
            "policies",
            "prompts",
            "raw_outputs",
            "accepted_artifacts",
            "output",
            "reports",
            "review",
        }
        _exact_keys(table, expected, "paths")
        validated: dict[str, str] = {}
        for name in sorted(expected):
            raw_path = _string(table[name], f"paths.{name}")
            assert isinstance(raw_path, str)
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts:
                raise PolicyBenchConfigError(f"paths.{name} must be a safe relative path")
            validated[name] = path.as_posix()
        return cls(**validated)


@dataclass(frozen=True, slots=True)
class PolicyBenchConfig:
    path: Path | None
    schema_version: str
    release_id: str
    taxonomy_version: str
    record_schema_version: str
    seed: int
    generated_at: str
    target_records: int
    languages: dict[str, float]
    domains: dict[str, int]
    category_quotas: dict[str, float]
    counterfactual_quotas: dict[str, float]
    generation: GenerationConfig
    quality: QualityConfig
    split_strategy: SplitStrategyConfig
    paths: OutputPaths
    review_sample_size: int

    @property
    def language_counts(self) -> dict[str, int]:
        return apportion_quota(self.target_records, self.languages, order=LANGUAGE_ORDER)

    @property
    def category_counts(self) -> dict[str, int]:
        return apportion_quota(self.target_records, self.category_quotas, order=CATEGORY_ORDER)

    @property
    def counterfactual_counts(self) -> dict[str, int]:
        share = sum(Decimal(str(value)) for value in self.counterfactual_quotas.values())
        total = int(
            (Decimal(self.target_records) * share).to_integral_value(rounding=ROUND_HALF_UP)
        )
        return apportion_quota(
            total,
            self.counterfactual_quotas,
            order=COUNTERFACTUAL_TYPE_ORDER,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("path", None)
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> PolicyBenchConfig:
        resolved = _resolve_environment(value, os.environ if environ is None else environ)
        root = _mapping(resolved, "config")
        _exact_keys(root, _ROOT_KEYS, "config")
        if root["schema_version"] != "0.1":
            raise PolicyBenchConfigError("schema_version must equal '0.1'")
        if root["taxonomy_version"] != "1.0":
            raise PolicyBenchConfigError("taxonomy_version must equal frozen version '1.0'")
        if root["record_schema_version"] != "0.1":
            raise PolicyBenchConfigError("record_schema_version must equal '0.1'")
        target = _integer(root["target_records"], "target_records", minimum=1)
        generated_at = _string(root["generated_at"], "generated_at")
        release_id = _string(root["release_id"], "release_id")
        assert isinstance(generated_at, str) and isinstance(release_id, str)
        try:
            parsed_timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise PolicyBenchConfigError("generated_at must be an RFC 3339 timestamp") from error
        if parsed_timestamp.tzinfo is None:
            raise PolicyBenchConfigError("generated_at must include an explicit UTC offset")

        domains = _quota_mapping(root["domains"], DOMAIN_ORDER, "domains", integer_values=True)
        assert all(isinstance(count, int) for count in domains.values())
        if sum(domains.values()) != target:
            raise PolicyBenchConfigError("domain counts must sum to target_records")
        languages = _quota_mapping(root["languages"], LANGUAGE_ORDER, "languages", require_sum=1.0)
        categories = _quota_mapping(
            root["category_quotas"],
            CATEGORY_ORDER,
            "category_quotas",
            require_sum=1.0,
        )
        counterfactuals = _quota_mapping(
            root["counterfactual_quotas"],
            COUNTERFACTUAL_TYPE_ORDER,
            "counterfactual_quotas",
        )
        if sum(counterfactuals.values()) > 1.0 + 1e-9:
            raise PolicyBenchConfigError("counterfactual_quotas cannot exceed 1.0 in total")
        review_sample_size = _integer(root["review_sample_size"], "review_sample_size", minimum=1)
        if review_sample_size > target:
            raise PolicyBenchConfigError("review_sample_size cannot exceed target_records")
        return cls(
            path=Path(path).resolve() if path is not None else None,
            schema_version="0.1",
            release_id=release_id,
            taxonomy_version="1.0",
            record_schema_version="0.1",
            seed=_integer(root["seed"], "seed"),
            generated_at=generated_at,
            target_records=target,
            languages={name: float(languages[name]) for name in LANGUAGE_ORDER},
            domains={name: int(domains[name]) for name in DOMAIN_ORDER},
            category_quotas={name: float(categories[name]) for name in CATEGORY_ORDER},
            counterfactual_quotas={
                name: float(counterfactuals[name]) for name in COUNTERFACTUAL_TYPE_ORDER
            },
            generation=GenerationConfig.from_mapping(root["generation"]),
            quality=QualityConfig.from_mapping(root["quality"]),
            split_strategy=SplitStrategyConfig.from_mapping(root["split_strategy"]),
            paths=OutputPaths.from_mapping(root["paths"]),
            review_sample_size=review_sample_size,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> PolicyBenchConfig:
        config_path = Path(path)
        try:
            raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"PolicyBench config not found: {config_path}") from error
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise PolicyBenchConfigError(
                f"cannot read PolicyBench config {config_path}: {error}"
            ) from error
        return cls.from_mapping(raw, path=config_path, environ=environ)


__all__ = [
    "CATEGORY_ORDER",
    "COUNTERFACTUAL_TYPE_ORDER",
    "DOMAIN_ORDER",
    "GenerationConfig",
    "LANGUAGE_ORDER",
    "OutputPaths",
    "PolicyBenchConfig",
    "PolicyBenchConfigError",
    "QualityConfig",
    "SPLIT_GROUP_FIELDS",
    "SplitStrategyConfig",
    "apportion_quota",
]
