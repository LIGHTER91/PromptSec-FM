"""Canonical PromptSec-PolicyBench record construction.

The builder treats provider realizations as untrusted linguistic data. Frozen
labels come only from a validated deterministic blueprint, while exact spans
are resolved locally over the final stored candidate text.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from promptsec.data.hashing import sha256_json, sha256_text
from promptsec.data.validation import derive_prompt_injection_verdict
from promptsec.policybench.spans import resolve_generation_anchors
from promptsec.policybench.validation import (
    require_valid_generation_response,
    require_valid_policybench_record,
)

GenerationMethod = Literal["DETERMINISTIC_TEMPLATE", "LLM_CONTROLLED"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANNOTATION_FIELDS = (
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "attack_families",
    "attack_objectives",
    "annotation_status",
)
_GROUPING_FIELDS = frozenset(
    {
        "policy_family",
        "scenario_template_family",
        "attack_template_family",
        "base_generation_family",
        "semantic_duplicate_cluster_id",
        "semantic_duplicate_cluster",
        "split_group_id",
    }
)
_SPLITS = frozenset(
    {
        "train",
        "validation",
        "test_policy_family_ood",
        "test_domain_ood",
        "test_language_ood",
        "test_counterfactual",
        "human_review_candidates",
    }
)
_DEFAULT_VALIDATION_CHECKS = (
    "STRUCTURE",
    "LANGUAGE",
    "DECLARED_ACTION",
    "SPANS",
    "AUTHORITY_CLAIM",
    "LABELS",
    "POLICY",
    "COUNTERFACTUAL",
    "DUPLICATION",
    "LABEL_LEAKAGE",
)


class PolicyBenchRecordError(ValueError):
    """Raised before schema validation when record inputs are malformed."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    raise PolicyBenchRecordError(f"{context} must be a mapping or object with to_dict()")


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyBenchRecordError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyBenchRecordError(f"{context} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PolicyBenchRecordError(f"{context} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class GenerationRecordMetadata:
    """Auditable generator metadata used by one accepted realization."""

    generator_provider: str
    generator_model: str
    generator_model_revision: str | None
    generation_prompt_version: str
    generation_seed: int
    generation_temperature: float
    generation_attempt: int
    generation_timestamp: str
    raw_generation_sha256: str | None = None
    failed_attempts: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    usage: Mapping[str, Any] | None = None
    generation_method: GenerationMethod = "LLM_CONTROLLED"
    validator_name: str = "promptsec-policybench"
    validator_version: str = "0.1"
    validation_timestamp: str | None = None
    validation_checks: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "generator_provider",
            "generator_model",
            "generation_prompt_version",
            "generation_timestamp",
            "validator_name",
            "validator_version",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if self.generator_model_revision is not None:
            _nonempty(self.generator_model_revision, "generator_model_revision")
        _integer(self.generation_seed, "generation_seed")
        _integer(self.generation_attempt, "generation_attempt", minimum=1)
        if (
            isinstance(self.generation_temperature, bool)
            or not isinstance(self.generation_temperature, (int, float))
            or not 0 <= float(self.generation_temperature) <= 2
        ):
            raise PolicyBenchRecordError("generation_temperature must be within [0, 2]")
        if self.raw_generation_sha256 is not None:
            _sha256(self.raw_generation_sha256, "raw_generation_sha256")
        if self.generation_method not in {"DETERMINISTIC_TEMPLATE", "LLM_CONTROLLED"}:
            raise PolicyBenchRecordError("unsupported generation_method")
        if self.validation_timestamp is not None:
            _nonempty(self.validation_timestamp, "validation_timestamp")
        if not isinstance(self.failed_attempts, tuple):
            raise PolicyBenchRecordError("failed_attempts must be a tuple")
        if not isinstance(self.validation_checks, tuple):
            raise PolicyBenchRecordError("validation_checks must be a tuple")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GenerationRecordMetadata:
        """Build typed metadata from the schema-aligned public field names."""

        if not isinstance(value, Mapping):
            raise PolicyBenchRecordError("generator_metadata must be an object")
        allowed = {
            "generator_provider",
            "generator_model",
            "generator_model_revision",
            "generation_prompt_version",
            "generation_seed",
            "generation_temperature",
            "generation_attempt",
            "generation_timestamp",
            "raw_generation_sha256",
            "failed_attempts",
            "usage",
            "generation_method",
            "validator_name",
            "validator_version",
            "validation_timestamp",
            "validation_checks",
        }
        unexpected = sorted(set(value).difference(allowed))
        if unexpected:
            raise PolicyBenchRecordError(f"generator_metadata has unexpected fields: {unexpected}")
        required = {
            "generator_provider",
            "generator_model",
            "generator_model_revision",
            "generation_prompt_version",
            "generation_seed",
            "generation_temperature",
            "generation_attempt",
            "generation_timestamp",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise PolicyBenchRecordError(f"generator_metadata is missing fields: {missing}")
        return cls(
            generator_provider=value["generator_provider"],
            generator_model=value["generator_model"],
            generator_model_revision=value["generator_model_revision"],
            generation_prompt_version=value["generation_prompt_version"],
            generation_seed=value["generation_seed"],
            generation_temperature=value["generation_temperature"],
            generation_attempt=value["generation_attempt"],
            generation_timestamp=value["generation_timestamp"],
            raw_generation_sha256=value.get("raw_generation_sha256"),
            failed_attempts=tuple(copy.deepcopy(value.get("failed_attempts", ()))),
            usage=copy.deepcopy(value.get("usage")),
            generation_method=value.get("generation_method", "LLM_CONTROLLED"),
            validator_name=value.get("validator_name", "promptsec-policybench"),
            validator_version=value.get("validator_version", "0.1"),
            validation_timestamp=value.get("validation_timestamp"),
            validation_checks=tuple(copy.deepcopy(value.get("validation_checks", ()))),
        )

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        prompt_version: str,
        seed: int,
        temperature: float,
        attempt: int = 1,
        failed_attempts: Sequence[Mapping[str, Any]] = (),
        generation_method: GenerationMethod | None = None,
    ) -> GenerationRecordMetadata:
        """Adapt a provider ``GenerationResponse`` without retaining raw text."""

        provider = _nonempty(getattr(response, "provider", None), "response.provider")
        raw_text = _nonempty(getattr(response, "raw_text", None), "response.raw_text")
        usage = getattr(response, "usage", None)
        usage_to_dict = getattr(usage, "to_dict", None)
        usage_value = usage_to_dict() if callable(usage_to_dict) else usage
        if isinstance(usage_value, Mapping) and not usage_value:
            usage_value = None
        return cls(
            generator_provider=provider,
            generator_model=_nonempty(getattr(response, "model", None), "response.model"),
            generator_model_revision=getattr(response, "model_revision", None),
            generation_prompt_version=prompt_version,
            generation_seed=seed,
            generation_temperature=temperature,
            generation_attempt=attempt,
            generation_timestamp=_nonempty(
                getattr(response, "generated_at", None), "response.generated_at"
            ),
            raw_generation_sha256=sha256_text(raw_text),
            failed_attempts=tuple(copy.deepcopy(failed_attempts)),
            usage=copy.deepcopy(usage_value),
            generation_method=(
                generation_method
                if generation_method is not None
                else "DETERMINISTIC_TEMPLATE"
                if provider == "mock"
                else "LLM_CONTROLLED"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_provider": self.generator_provider,
            "generator_model": self.generator_model,
            "generator_model_revision": self.generator_model_revision,
            "generation_prompt_version": self.generation_prompt_version,
            "generation_seed": self.generation_seed,
            "generation_temperature": float(self.generation_temperature),
            "generation_attempt": self.generation_attempt,
            "generation_timestamp": self.generation_timestamp,
            "raw_generation_sha256": self.raw_generation_sha256,
            "failed_attempts": copy.deepcopy(list(self.failed_attempts)),
            "usage": copy.deepcopy(self.usage),
            "generation_method": self.generation_method,
            "validator_name": self.validator_name,
            "validator_version": self.validator_version,
            "validation_timestamp": self.validation_timestamp,
            "validation_checks": copy.deepcopy(list(self.validation_checks)),
        }


def _generator_metadata(value: Any) -> GenerationRecordMetadata:
    if isinstance(value, GenerationRecordMetadata):
        return value
    mapping = _mapping(value, "generator_metadata")
    return GenerationRecordMetadata.from_mapping(mapping)


def _usage(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    usage = _mapping(value, "usage")
    allowed = {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "invocation_duration_seconds",
        "batch_id",
        "batch_size",
        "batch_position",
        "exit_status",
    }
    unexpected = sorted(set(usage).difference(allowed))
    if unexpected:
        raise PolicyBenchRecordError(f"usage has unexpected fields: {unexpected}")
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    input_tokens = _integer(input_tokens, "usage.input_tokens")
    output_tokens = _integer(output_tokens, "usage.output_tokens")
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    total_tokens = _integer(total_tokens, "usage.total_tokens")
    cost = usage.get("cost_usd")
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0
    ):
        raise PolicyBenchRecordError("usage.cost_usd must be a non-negative number or null")
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": float(cost) if cost is not None else None,
    }
    for name in ("cached_input_tokens", "reasoning_output_tokens", "exit_status"):
        if name in usage:
            result[name] = _integer(usage[name], f"usage.{name}")
    if "invocation_duration_seconds" in usage:
        duration = usage["invocation_duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise PolicyBenchRecordError(
                "usage.invocation_duration_seconds must be a non-negative number"
            )
        result["invocation_duration_seconds"] = float(duration)
    if "batch_id" in usage:
        result["batch_id"] = _nonempty(usage["batch_id"], "usage.batch_id")
    if "batch_size" in usage:
        result["batch_size"] = _integer(usage["batch_size"], "usage.batch_size", minimum=1)
    if "batch_position" in usage:
        result["batch_position"] = _integer(usage["batch_position"], "usage.batch_position")
        if "batch_size" not in result or result["batch_position"] >= result["batch_size"]:
            raise PolicyBenchRecordError("usage.batch_position must be within batch_size")
    return result


def _failed_attempts(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {
        "generation_attempt",
        "generation_timestamp",
        "raw_generation_sha256",
        "rejection_reasons",
    }
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        attempt = _mapping(value, f"failed_attempts[{index}]")
        if set(attempt) != expected:
            raise PolicyBenchRecordError(
                f"failed_attempts[{index}] must contain exactly {sorted(expected)}"
            )
        result.append(attempt)
    return result


def _validation_checks(
    metadata: GenerationRecordMetadata, counterfactual: Any
) -> list[dict[str, Any]]:
    if metadata.validation_checks:
        return [
            _mapping(value, f"validation_checks[{index}]")
            for index, value in enumerate(metadata.validation_checks)
        ]
    return [
        {
            "name": name,
            "status": (
                "SKIPPED"
                if name == "DUPLICATION" or (name == "COUNTERFACTUAL" and counterfactual is None)
                else "PASS"
            ),
            "detail": None,
        }
        for name in _DEFAULT_VALIDATION_CHECKS
    ]


def _grouping(
    grouping: Mapping[str, Any] | None,
    *,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    candidate_text: str,
) -> dict[str, Any]:
    supplied = {} if grouping is None else _mapping(grouping, "grouping")
    unexpected = sorted(set(supplied).difference(_GROUPING_FIELDS))
    if unexpected:
        raise PolicyBenchRecordError(f"grouping has unexpected fields: {unexpected}")

    counterfactual = plan.get("counterfactual")
    counterfactual = counterfactual if isinstance(counterfactual, Mapping) else None
    scenario_id = str(plan["scenario_id"])
    domain = str(plan["domain"])
    category = str(plan["category"])
    default_base = (
        str(counterfactual["parent_scenario_id"]) if counterfactual is not None else scenario_id
    )
    split_identity = {"scenario_id": scenario_id, "policy_id": plan.get("policy_id")}
    default_split_group = (
        str(counterfactual["counterfactual_group_id"])
        if counterfactual is not None
        else f"base_{sha256_json(split_identity)[:24]}"
    )
    semantic = supplied.get(
        "semantic_duplicate_cluster_id",
        supplied.get(
            "semantic_duplicate_cluster", f"semantic_pending_{sha256_text(candidate_text)[:24]}"
        ),
    )
    return {
        "policy_family": supplied.get("policy_family", policy["policy_family"]),
        "scenario_template_family": supplied.get(
            "scenario_template_family", f"{domain}:{category.lower()}"
        ),
        "attack_template_family": supplied.get("attack_template_family"),
        "base_generation_family": supplied.get("base_generation_family", default_base),
        "semantic_duplicate_cluster_id": semantic,
        "split_group_id": supplied.get("split_group_id", default_split_group),
    }


def _record_id(plan: Mapping[str, Any], index: int) -> str:
    """Return an ID independent of labels, confidence, and generator identity."""

    identity = {
        "schema_version": "0.1",
        "scenario_id": plan["scenario_id"],
        "policy_id": plan["policy_id"],
        "domain": plan["domain"],
        "language": plan["language"],
        "stable_index": index,
    }
    return f"pspb_{sha256_json(identity)[:24]}"


def _policy_rule_ids(policy: Mapping[str, Any]) -> set[str]:
    rule_ids: set[str] = set()
    for field_name in (
        "structured_rules",
        "confirmation_requirements",
        "source_authority_rules",
    ):
        values = policy.get(field_name)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, Mapping) and isinstance(value.get("rule_id"), str):
                rule_ids.add(value["rule_id"])
    return rule_ids


def build_policybench_record(
    blueprint: Any,
    realization: Mapping[str, Any],
    policy: Mapping[str, Any],
    generator_metadata: GenerationRecordMetadata | Mapping[str, Any],
    *,
    config_path: str | Path,
    config_sha256: str,
    accepted_artifact_path: str | Path,
    index: int,
    grouping: Mapping[str, Any] | None = None,
    split: str | None = None,
    dataset_split: str | None = None,
) -> dict[str, Any]:
    """Build and fully validate one canonical SILVER PolicyBench record."""

    plan = _mapping(blueprint, "blueprint")
    response = _mapping(realization, "realization")
    policy_value = _mapping(policy, "policy")
    metadata = _generator_metadata(generator_metadata)
    stable_index = _integer(index, "index")
    config_digest = _sha256(config_sha256, "config_sha256")
    raw_artifact = Path(accepted_artifact_path)
    if raw_artifact.is_absolute() or ".." in raw_artifact.parts or not raw_artifact.name:
        raise PolicyBenchRecordError(
            "accepted_artifact_path must be a safe repository-relative path"
        )
    raw_artifact_value = raw_artifact.as_posix()
    if split is not None and dataset_split is not None and split != dataset_split:
        raise PolicyBenchRecordError("split and dataset_split disagree")
    selected_split = dataset_split or split or "train"
    if selected_split not in _SPLITS:
        raise PolicyBenchRecordError(f"unsupported dataset split: {selected_split!r}")

    require_valid_generation_response(response, plan, policy_value)
    if plan.get("policy_id") != policy_value.get("policy_id"):
        raise PolicyBenchRecordError("blueprint policy_id does not match policy")
    if plan.get("domain") != policy_value.get("domain"):
        raise PolicyBenchRecordError("blueprint domain does not match policy")
    unknown_rule_ids = sorted(
        set(plan["policy_rule_ids"]).difference(_policy_rule_ids(policy_value))
    )
    if unknown_rule_ids:
        raise PolicyBenchRecordError(
            f"blueprint references policy rule IDs absent from policy: {unknown_rule_ids}"
        )

    expected = plan["expected_annotations"]
    if not isinstance(expected, Mapping):
        raise PolicyBenchRecordError("blueprint expected_annotations must be an object")
    candidate_text = response["candidate_text"]
    spans = resolve_generation_anchors(
        candidate_text,
        directive_anchors=response["directive_anchors"],
        injection_payload_anchors=response["injection_payload_anchors"],
        authority_claim_anchors=response["authority_claim_anchors"],
    )
    annotations = {
        field_name: copy.deepcopy(expected[field_name]) for field_name in _ANNOTATION_FIELDS
    }
    annotations["spans"] = [span.to_dict() for span in spans]
    # This field is frozen and required, but generated records have no human
    # annotator. Zero is a sentinel, never model or blueprint confidence.
    annotations["annotator_confidence"] = 0.0
    verdict = derive_prompt_injection_verdict(annotations)

    blueprint_hash = sha256_json(plan)
    policy_hash = sha256_json(policy_value)
    raw_generation_hash = metadata.raw_generation_sha256 or sha256_json(response)
    grouped = _grouping(
        grouping,
        plan=plan,
        policy=policy_value,
        candidate_text=candidate_text,
    )
    counterfactual = copy.deepcopy(plan.get("counterfactual"))
    is_counterfactual_sibling = bool(
        isinstance(counterfactual, Mapping)
        and plan["scenario_id"] != counterfactual.get("parent_scenario_id")
    )
    counterfactual_type = (
        counterfactual.get("counterfactual_type") if isinstance(counterfactual, Mapping) else None
    )
    parent_artifact_value = None
    if is_counterfactual_sibling:
        parent_artifact_value = (
            raw_artifact.parent / f"{counterfactual['parent_scenario_id']}.json"
        ).as_posix()
    candidate_from_parent = bool(
        is_counterfactual_sibling and counterfactual_type != "PRESENTATION_CHANGE"
    )
    policy_from_parent = bool(
        is_counterfactual_sibling
        and counterfactual_type not in {"POLICY_CHANGE", "AUTHORITY_DELEGATION_CHANGE"}
    )
    goal_from_parent = bool(is_counterfactual_sibling and counterfactual_type != "USER_GOAL_CHANGE")
    checks = _validation_checks(metadata, counterfactual)
    if any(check.get("status") == "FAIL" for check in checks):
        raise PolicyBenchRecordError("SILVER_VALIDATED record cannot contain a failed check")

    config_value = Path(config_path).as_posix()
    original_fields = copy.deepcopy(response)
    record = {
        "id": _record_id(plan, stable_index),
        "taxonomy_version": "1.0",
        "context": {
            "protected_policy": response["protected_policy"],
            "user_goal": response["user_goal"],
            "available_capabilities": copy.deepcopy(plan["available_capabilities"]),
        },
        "content": {
            "text": candidate_text,
            "language": plan["language"],
            "delivery_mode": plan["delivery_mode"],
            "source_role": plan["source_role"],
            "content_origin": plan["content_origin"],
            "ingestion_path": plan["ingestion_path"],
            "modality": plan["modality"],
            "source_integrity": plan["source_integrity"],
        },
        "annotations": annotations,
        "derived": {"prompt_injection_verdict": verdict},
        "extensions": {
            "policybench_v0_1": {
                "schema_version": "0.1",
                "data_quality": "SILVER_VALIDATED",
                "human_validation_status": "PENDING",
                "generation_method": metadata.generation_method,
                "policy": {
                    "policy_id": policy_value["policy_id"],
                    "policy_family": policy_value["policy_family"],
                    "policy_version": policy_value["policy_version"],
                    "matched_rule_ids": copy.deepcopy(plan["policy_rule_ids"]),
                    "policy_sha256": policy_hash,
                    "effective_policy_sha256": sha256_text(response["protected_policy"] or ""),
                    "authority_delegation_override": copy.deepcopy(
                        plan.get("authority_delegation_override")
                    ),
                },
                "blueprint": {
                    "scenario_blueprint_id": plan["scenario_id"],
                    "category": plan["category"],
                    "domain": plan["domain"],
                    "language": plan["language"],
                    "scenario_template_family": grouped["scenario_template_family"],
                    "attack_template_family": grouped["attack_template_family"],
                    "base_generation_family": grouped["base_generation_family"],
                    "blueprint_sha256": blueprint_hash,
                },
                "counterfactual": counterfactual,
                "generation": {
                    "generator_provider": metadata.generator_provider,
                    "generator_model": metadata.generator_model,
                    "generator_model_revision": metadata.generator_model_revision,
                    "generation_prompt_version": metadata.generation_prompt_version,
                    "generation_seed": metadata.generation_seed,
                    "generation_temperature": float(metadata.generation_temperature),
                    "generation_attempt": metadata.generation_attempt,
                    "generation_timestamp": metadata.generation_timestamp,
                    "raw_generation_sha256": raw_generation_hash,
                    "blueprint_sha256": blueprint_hash,
                    "policy_sha256": policy_hash,
                    "validation_status": "PASSED",
                    "failed_attempts": _failed_attempts(metadata.failed_attempts),
                    "usage": _usage(metadata.usage),
                },
                "validation": {
                    "validator_name": metadata.validator_name,
                    "validator_version": metadata.validator_version,
                    "validated_at": metadata.validation_timestamp or metadata.generation_timestamp,
                    "overall_status": "PASSED",
                    "checks": checks,
                    "rejection_reasons": [],
                },
                "grouping": grouped,
                "dataset_split": selected_split,
            }
        },
        "metadata": {
            "record_schema_version": "0.1",
            "dataset_provenance": {
                "source_dataset": {
                    "id": "promptsec-policybench-v0.1",
                    "name": "PromptSec-PolicyBench",
                    "version": "0.1",
                    "revision": config_digest,
                    "url": "https://github.com/LIGHTER91/PromptSec-FM",
                    "license_manifest": "manifests/policybench-v0.1.json",
                },
                "source_record": {
                    "id": plan["scenario_id"],
                    "split": selected_split,
                    "index": stable_index,
                    "raw_artifact": raw_artifact_value,
                    "raw_record_sha256": sha256_json(original_fields),
                    "original_fields": original_fields,
                    "original_labels": copy.deepcopy(expected),
                },
                "import": {
                    "importer": "PolicyBenchGenerator",
                    "importer_version": "0.1",
                    "config": config_value,
                    "config_sha256": config_digest,
                    "imported_at": metadata.generation_timestamp,
                    "transformations": [
                        "resolve_exact_anchors_to_python_codepoint_spans",
                        "copy_deterministic_blueprint_labels",
                        "force_generated_annotator_confidence_zero",
                        "derive_prompt_injection_verdict",
                    ]
                    + (
                        ["copy_candidate_from_parent_counterfactual_accepted_artifact"]
                        if candidate_from_parent
                        else []
                    )
                    + (
                        ["copy_policy_from_parent_counterfactual_accepted_artifact"]
                        if policy_from_parent
                        else []
                    )
                    + (
                        ["copy_user_goal_from_parent_counterfactual_accepted_artifact"]
                        if goal_from_parent
                        else []
                    ),
                },
                "mapping": {
                    "ruleset": "policybench_blueprint_v0.1",
                    "status": "NEEDS_REVIEW",
                    "field_mappings": [
                        {
                            "source": (
                                f"{parent_artifact_value}#/realization/candidate_text"
                                if candidate_from_parent
                                else "candidate_text"
                            ),
                            "target": "content.text",
                            "method": (
                                "deterministic_parent_artifact_copy"
                                if candidate_from_parent
                                else "exact_provider_realization"
                            ),
                        },
                        {
                            "source": (
                                f"{parent_artifact_value}#/realization/protected_policy"
                                if policy_from_parent
                                else "protected_policy"
                            ),
                            "target": "context.protected_policy",
                            "method": (
                                "deterministic_parent_artifact_copy"
                                if policy_from_parent
                                else "validated_blueprint_policy_realization"
                            ),
                        },
                        {
                            "source": (
                                f"{parent_artifact_value}#/realization/user_goal"
                                if goal_from_parent
                                else "user_goal"
                            ),
                            "target": "context.user_goal",
                            "method": (
                                "deterministic_parent_artifact_copy"
                                if goal_from_parent
                                else "validated_provider_realization"
                            ),
                        },
                        {
                            "source": "expected_annotations",
                            "target": "annotations",
                            "method": "deterministic_scenario_blueprint",
                        },
                    ],
                    "unmapped_labels": [],
                    "review_reasons": [
                        "AI-generated SILVER record requires human double annotation"
                    ],
                },
                "checksums": {
                    "source_text_sha256": sha256_text(candidate_text),
                    "canonical_text_sha256": sha256_text(candidate_text),
                },
            },
        },
    }
    require_valid_policybench_record(record, plan)
    return record


__all__ = [
    "GenerationMethod",
    "GenerationRecordMetadata",
    "PolicyBenchRecordError",
    "build_policybench_record",
]
