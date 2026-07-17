"""Leakage-resistant, transitive splitting for PromptSec-PolicyBench."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_json, sha256_text
from promptsec.policybench.deduplication import DuplicateAnalysis
from promptsec.policybench.io import write_json, write_jsonl, write_named_checksums

PUBLISHED_SPLITS = (
    "train",
    "validation",
    "test_policy_family_ood",
    "test_domain_ood",
    "test_language_ood",
    "test_counterfactual",
    "human_review_candidates",
)
ALL_SPLITS = (*PUBLISHED_SPLITS, "EXCLUDED")
_GROUP_FIELDS = (
    "policy_family",
    "scenario_template_family",
    "attack_template_family",
    "counterfactual_group_id",
    "base_generation_family",
    "semantic_duplicate_cluster",
)
_LANGUAGE_STRATIFIED_FIELDS = frozenset(
    {
        "policy_family",
        "scenario_template_family",
        "attack_template_family",
        "base_generation_family",
    }
)
_DEFAULT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test_policy_family_ood": 0.04,
    "test_domain_ood": 0.03,
    "test_language_ood": 0.03,
    "test_counterfactual": 0.05,
}


class PolicyBenchSplitError(ValueError):
    """Raised when leakage constraints cannot produce the requested partition."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        self.diagnostics = dict(diagnostics or {})
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PolicyBenchSplitResult:
    """Updated records, materialized split views, and a text-free split report."""

    assignments: dict[str, str]
    records: tuple[dict[str, Any], ...]
    records_by_split: dict[str, tuple[dict[str, Any], ...]]
    report: dict[str, Any]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


@dataclass(frozen=True, slots=True)
class _Component:
    group_id: str
    member_ids: tuple[str, ...]
    domains: frozenset[str]
    languages: frozenset[str]
    has_counterfactual: bool

    @property
    def size(self) -> int:
        return len(self.member_ids)


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(value, Mapping):
        raise PolicyBenchSplitError(
            f"record {record.get('id')!r}: extensions.policybench_v0_1 must be an object"
        )
    for field in ("policy", "blueprint", "generation", "validation", "grouping"):
        if not isinstance(value.get(field), Mapping):
            raise PolicyBenchSplitError(
                f"record {record.get('id')!r}: policybench extension {field} must be an object"
            )
    return value


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyBenchSplitError(f"{context} must be a non-empty string")
    return value


def _duplicate_mapping(
    value: DuplicateAnalysis | Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, DuplicateAnalysis):
        return value.assignments
    if not isinstance(value, Mapping):
        raise PolicyBenchSplitError("duplicate assignments must be a mapping")
    return value


def _group_values(
    record: Mapping[str, Any],
    duplicate_assignment: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    record_id = _required_string(record.get("id"), "record.id")
    extension = _extension(record)
    policy = extension["policy"]
    blueprint = extension["blueprint"]
    grouping = extension["grouping"]
    assert isinstance(policy, Mapping)
    assert isinstance(blueprint, Mapping)
    assert isinstance(grouping, Mapping)
    counterfactual = extension.get("counterfactual")
    counterfactual_id = None
    if counterfactual is not None:
        if not isinstance(counterfactual, Mapping):
            raise PolicyBenchSplitError(
                f"record {record_id!r}: counterfactual must be an object or null"
            )
        counterfactual_id = _required_string(
            counterfactual.get("counterfactual_group_id"),
            f"record {record_id!r} counterfactual_group_id",
        )
    semantic_cluster = (
        duplicate_assignment.get("semantic_duplicate_cluster_id")
        if isinstance(duplicate_assignment, Mapping)
        else grouping.get("semantic_duplicate_cluster_id")
    )
    attack_family = blueprint.get("attack_template_family")
    if attack_family is not None:
        attack_family = _required_string(
            attack_family, f"record {record_id!r} attack_template_family"
        )
    return {
        "policy_family": _required_string(
            policy.get("policy_family"), f"record {record_id!r} policy_family"
        ),
        "scenario_template_family": _required_string(
            blueprint.get("scenario_template_family"),
            f"record {record_id!r} scenario_template_family",
        ),
        "attack_template_family": attack_family,
        "counterfactual_group_id": counterfactual_id,
        "base_generation_family": _required_string(
            blueprint.get("base_generation_family"),
            f"record {record_id!r} base_generation_family",
        ),
        "semantic_duplicate_cluster": _required_string(
            semantic_cluster,
            f"record {record_id!r} semantic_duplicate_cluster",
        ),
    }


def _record_language(record: Mapping[str, Any]) -> str:
    record_id = _required_string(record.get("id"), "record.id")
    blueprint = _extension(record)["blueprint"]
    assert isinstance(blueprint, Mapping)
    return _required_string(blueprint.get("language"), f"record {record_id!r} blueprint.language")


def _linkage_value(field: str, value: str, language: str) -> str:
    """Namespace template/family leakage within language without changing records."""

    return f"{language}::{value}" if field in _LANGUAGE_STRATIFIED_FIELDS else value


def _validated_ratios(value: Mapping[str, float] | None) -> dict[str, float]:
    source = _DEFAULT_RATIOS if value is None else value
    if set(source) != set(_DEFAULT_RATIOS):
        raise PolicyBenchSplitError(f"ratios must contain exactly {sorted(_DEFAULT_RATIOS)}")
    result: dict[str, float] = {}
    for name in _DEFAULT_RATIOS:
        item = source[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PolicyBenchSplitError(f"ratio {name} must be a number")
        number = float(item)
        if not 0.0 <= number <= 1.0:
            raise PolicyBenchSplitError(f"ratio {name} must be within [0, 1]")
        result[name] = number
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise PolicyBenchSplitError("split ratios must sum to 1.0")
    return result


def _rank(group_id: str, *, seed: int, salt: str) -> str:
    return sha256_text(f"{seed}:{salt}:{group_id}")


def _take_to_target(
    components: list[_Component],
    *,
    target_records: int,
    seed: int,
    salt: str,
) -> tuple[list[_Component], list[_Component]]:
    ordered = sorted(
        components,
        key=lambda item: (_rank(item.group_id, seed=seed, salt=salt), item.group_id),
    )
    if target_records <= 0 or not ordered:
        return [], ordered
    selected: list[_Component] = []
    count = 0
    for component in ordered:
        if count >= target_records:
            break
        selected.append(component)
        count += component.size
    selected_ids = {component.group_id for component in selected}
    remaining = [component for component in ordered if component.group_id not in selected_ids]
    return selected, remaining


def _mark_excluded(record: dict[str, Any], reason: str) -> None:
    extension = record["extensions"]["policybench_v0_1"]
    extension["data_quality"] = "EXCLUDED"
    extension["dataset_split"] = "EXCLUDED"
    generation = extension["generation"]
    generation["validation_status"] = "FAILED"
    validation = extension["validation"]
    validation["overall_status"] = "FAILED"
    if validation.get("validated_at") is None:
        validation["validated_at"] = generation["generation_timestamp"]
    checks = [
        check
        for check in validation.get("checks", [])
        if not isinstance(check, Mapping) or check.get("name") != "DUPLICATION"
    ]
    checks.append({"name": "DUPLICATION", "status": "FAIL", "detail": reason})
    validation["checks"] = checks
    rejection_reasons = set(validation.get("rejection_reasons", []))
    rejection_reasons.add(reason)
    validation["rejection_reasons"] = sorted(rejection_reasons)


def _requested_splits(
    ratios: Mapping[str, float],
    *,
    held_out_domain: str | None,
    held_out_language: str | None,
    human_review_records: int,
) -> set[str]:
    requested = {
        split
        for split, ratio in ratios.items()
        if ratio > 0
        and (split != "test_domain_ood" or held_out_domain is not None)
        and (split != "test_language_ood" or held_out_language is not None)
    }
    if human_review_records > 0:
        requested.add("human_review_candidates")
    return requested


def assign_policybench_splits(
    records: Iterable[Mapping[str, Any]],
    duplicate_assignments: DuplicateAnalysis | Mapping[str, Mapping[str, Any]] | None = None,
    *,
    seed: int = 0,
    ratios: Mapping[str, float] | None = None,
    held_out_domain: str | None = None,
    held_out_language: str | None = None,
    human_review_records: int = 0,
    maximum_component_share: float = 0.50,
    minimum_components_per_requested_split: int = 2,
    require_populated_splits: bool = True,
) -> PolicyBenchSplitResult:
    """Create an exclusive split partition after transitive leakage closure.

    Priority is language OOD, domain OOD, counterfactual test, policy-family OOD,
    human review, validation, then train.  Configured OOD ratios are reporting
    targets only: every component containing the held-out language/domain is kept
    out of train, even when this makes the achieved OOD share larger.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PolicyBenchSplitError("seed must be an integer")
    if not isinstance(human_review_records, int) or isinstance(human_review_records, bool):
        raise PolicyBenchSplitError("human_review_records must be an integer")
    if human_review_records < 0:
        raise PolicyBenchSplitError("human_review_records cannot be negative")
    if (
        isinstance(minimum_components_per_requested_split, bool)
        or not isinstance(minimum_components_per_requested_split, int)
        or minimum_components_per_requested_split < 1
    ):
        raise PolicyBenchSplitError("minimum_components_per_requested_split must be positive")
    if (
        isinstance(maximum_component_share, bool)
        or not isinstance(maximum_component_share, (int, float))
        or not 0 < float(maximum_component_share) <= 1
    ):
        raise PolicyBenchSplitError("maximum_component_share must be within (0, 1]")
    split_ratios = _validated_ratios(ratios)
    duplicates = _duplicate_mapping(duplicate_assignments)

    records_by_id: dict[str, Mapping[str, Any]] = {}
    values_by_id: dict[str, dict[str, str | None]] = {}
    language_by_id: dict[str, str] = {}
    rejected_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise PolicyBenchSplitError("every record must be an object")
        record_id = _required_string(record.get("id"), "record.id")
        if record_id in records_by_id:
            raise PolicyBenchSplitError(f"duplicate canonical record id: {record_id}")
        assignment = duplicates.get(record_id)
        if duplicate_assignments is not None and not isinstance(assignment, Mapping):
            raise PolicyBenchSplitError(f"duplicate assignment missing record id {record_id!r}")
        records_by_id[record_id] = record
        values_by_id[record_id] = _group_values(record, assignment)
        language_by_id[record_id] = _record_language(record)
        if isinstance(assignment, Mapping) and str(assignment.get("decision", "")).startswith(
            "REJECT_"
        ):
            rejected_ids.add(record_id)

    active_ids = sorted(set(records_by_id).difference(rejected_ids))
    if not active_ids:
        raise PolicyBenchSplitError("no non-excluded records are available for splitting")
    union = _UnionFind(active_ids)
    linkage: dict[tuple[str, str], str] = {}
    linkage_counts: Counter[str] = Counter()
    for record_id in active_ids:
        for field in _GROUP_FIELDS:
            value = values_by_id[record_id][field]
            if value is None:
                continue
            key = (field, _linkage_value(field, value, language_by_id[record_id]))
            previous = linkage.get(key)
            if previous is not None:
                union.union(previous, record_id)
                linkage_counts[field] += 1
            else:
                linkage[key] = record_id

    component_members: dict[str, list[str]] = defaultdict(list)
    for record_id in active_ids:
        component_members[union.find(record_id)].append(record_id)
    components: list[_Component] = []
    for member_ids in component_members.values():
        ordered_ids = tuple(sorted(member_ids))
        group_id = f"split_group_{sha256_json(list(ordered_ids))[:24]}"
        domains = frozenset(
            _required_string(
                _extension(records_by_id[record_id])["blueprint"].get("domain"),
                f"record {record_id!r} blueprint.domain",
            )
            for record_id in ordered_ids
        )
        languages = frozenset(
            _required_string(
                _extension(records_by_id[record_id])["blueprint"].get("language"),
                f"record {record_id!r} blueprint.language",
            )
            for record_id in ordered_ids
        )
        components.append(
            _Component(
                group_id=group_id,
                member_ids=ordered_ids,
                domains=domains,
                languages=languages,
                has_counterfactual=any(
                    values_by_id[record_id]["counterfactual_group_id"] is not None
                    for record_id in ordered_ids
                ),
            )
        )
    components.sort(key=lambda item: item.group_id)
    largest = max(component.size for component in components)
    largest_share = largest / len(active_ids)
    collapse_diagnostics = {
        "components": len(components),
        "largest_component_records": largest,
        "largest_component_share": round(largest_share, 6),
        "maximum_component_share": float(maximum_component_share),
    }
    requested = _requested_splits(
        split_ratios,
        held_out_domain=held_out_domain,
        held_out_language=held_out_language,
        human_review_records=human_review_records,
    )
    if len(requested) > 1 and largest_share > float(maximum_component_share):
        raise PolicyBenchSplitError(
            "transitive leakage grouping collapsed too much of the corpus into one component",
            diagnostics=collapse_diagnostics,
        )

    component_split: dict[str, str] = {}
    remaining = list(components)

    if held_out_language is not None:
        selected = [item for item in remaining if held_out_language in item.languages]
        for component in selected:
            component_split[component.group_id] = "test_language_ood"
        selected_ids = {item.group_id for item in selected}
        remaining = [item for item in remaining if item.group_id not in selected_ids]

    if held_out_domain is not None:
        selected = [item for item in remaining if held_out_domain in item.domains]
        for component in selected:
            component_split[component.group_id] = "test_domain_ood"
        selected_ids = {item.group_id for item in selected}
        remaining = [item for item in remaining if item.group_id not in selected_ids]

    total_active = len(active_ids)
    counterfactual_candidates = [item for item in remaining if item.has_counterfactual]
    counterfactual_target = round(total_active * split_ratios["test_counterfactual"])
    if split_ratios["test_counterfactual"] > 0:
        counterfactual_target = max(1, counterfactual_target)
    selected, _unused = _take_to_target(
        counterfactual_candidates,
        target_records=counterfactual_target,
        seed=seed,
        salt="counterfactual",
    )
    for component in selected:
        component_split[component.group_id] = "test_counterfactual"
    selected_ids = {item.group_id for item in selected}
    remaining = [item for item in remaining if item.group_id not in selected_ids]

    policy_target = round(total_active * split_ratios["test_policy_family_ood"])
    if split_ratios["test_policy_family_ood"] > 0:
        policy_target = max(1, policy_target)
    selected, remaining = _take_to_target(
        remaining,
        target_records=policy_target,
        seed=seed,
        salt="policy-family-ood",
    )
    for component in selected:
        component_split[component.group_id] = "test_policy_family_ood"

    selected, remaining = _take_to_target(
        remaining,
        target_records=human_review_records,
        seed=seed,
        salt="human-review",
    )
    for component in selected:
        component_split[component.group_id] = "human_review_candidates"

    general_ratio = split_ratios["train"] + split_ratios["validation"]
    if general_ratio <= 0 and remaining:
        raise PolicyBenchSplitError("train and validation ratios leave records unassigned")
    validation_target = (
        round(sum(item.size for item in remaining) * split_ratios["validation"] / general_ratio)
        if general_ratio > 0
        else 0
    )
    if split_ratios["validation"] > 0 and remaining:
        validation_target = max(1, validation_target)
    selected, remaining = _take_to_target(
        remaining,
        target_records=validation_target,
        seed=seed,
        salt="validation",
    )
    for component in selected:
        component_split[component.group_id] = "validation"
    for component in remaining:
        component_split[component.group_id] = "train"

    assignments: dict[str, str] = {record_id: "EXCLUDED" for record_id in rejected_ids}
    split_group_by_id: dict[str, str] = {}
    for component in components:
        split = component_split[component.group_id]
        for record_id in component.member_ids:
            assignments[record_id] = split
            split_group_by_id[record_id] = component.group_id
    for record_id in rejected_ids:
        split_group_by_id[record_id] = f"split_group_excluded_{sha256_text(record_id)[:24]}"
    assignments = dict(sorted(assignments.items()))

    counts = Counter(assignments.values())
    component_counts = Counter(component_split.values())
    missing_requested = sorted(split for split in requested if counts.get(split, 0) == 0)
    component_capacity_floor = max(1, len(components) // max(1, len(requested)))
    effective_minimum_components = (
        min(minimum_components_per_requested_split, component_capacity_floor)
        if len(active_ids) >= 1_000
        else 1
    )
    underpowered_splits = sorted(
        split
        for split in requested
        if component_counts.get(split, 0) < effective_minimum_components
    )
    configured_record_targets = {
        "test_counterfactual": counterfactual_target,
        "test_policy_family_ood": policy_target,
        "human_review_candidates": human_review_records,
        "validation": validation_target,
    }
    target_deviations = {
        split: {
            "target_records": target,
            "achieved_records": counts.get(split, 0),
            "absolute_deviation": abs(counts.get(split, 0) - target),
            "within_one_component": abs(counts.get(split, 0) - target) <= largest,
        }
        for split, target in configured_record_targets.items()
        if split in requested
    }
    excessive_target_deviation = sorted(
        split for split, values in target_deviations.items() if not values["within_one_component"]
    )
    if require_populated_splits and (
        missing_requested or underpowered_splits or excessive_target_deviation
    ):
        diagnostics = {
            **collapse_diagnostics,
            "requested_splits": sorted(requested),
            "records_by_split": {split: counts.get(split, 0) for split in ALL_SPLITS},
            "components_by_split": {
                split: component_counts.get(split, 0) for split in PUBLISHED_SPLITS
            },
            "minimum_components_per_requested_split": effective_minimum_components,
            "missing_requested_splits": missing_requested,
            "underpowered_requested_splits": underpowered_splits,
            "target_deviations": target_deviations,
            "excessive_target_deviation_splits": excessive_target_deviation,
        }
        raise PolicyBenchSplitError(
            "requested splits failed population, independent-component, or target-deviation guards",
            diagnostics=diagnostics,
        )

    updated_records: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ALL_SPLITS}
    for record_id in sorted(records_by_id):
        record = copy.deepcopy(records_by_id[record_id])
        extension = record["extensions"]["policybench_v0_1"]
        values = values_by_id[record_id]
        extension["grouping"] = {
            "policy_family": values["policy_family"],
            "scenario_template_family": values["scenario_template_family"],
            "attack_template_family": values["attack_template_family"],
            "base_generation_family": values["base_generation_family"],
            "semantic_duplicate_cluster_id": values["semantic_duplicate_cluster"],
            "split_group_id": split_group_by_id[record_id],
        }
        split = assignments[record_id]
        extension["dataset_split"] = split
        provenance = record.get("metadata", {}).get("dataset_provenance", {})
        source_record = provenance.get("source_record") if isinstance(provenance, Mapping) else None
        if isinstance(source_record, dict):
            source_record["split"] = split
        if split == "EXCLUDED":
            duplicate = duplicates.get(record_id, {})
            decision = duplicate.get("decision", "REJECT_DUPLICATE")
            _mark_excluded(record, f"duplicate policy decision: {decision}")
        updated_records.append(record)
        by_split[split].append(record)

    value_splits: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in _GROUP_FIELDS
    }
    raw_value_splits: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in _GROUP_FIELDS
    }
    for record_id, split in assignments.items():
        if split == "EXCLUDED":
            continue
        for field, value in values_by_id[record_id].items():
            if value is not None:
                linkage_value = _linkage_value(field, value, language_by_id[record_id])
                value_splits[field][linkage_value].add(split)
                raw_value_splits[field][value].add(split)
    leakage_by_field = {
        field: sorted(value for value, splits in values.items() if len(splits) > 1)
        for field, values in value_splits.items()
    }
    raw_leakage_by_field = {
        field: sorted(value for value, splits in values.items() if len(splits) > 1)
        for field, values in raw_value_splits.items()
    }
    policy_ood_families = {
        str(values_by_id[record_id]["policy_family"])
        for record_id, split in assignments.items()
        if split == "test_policy_family_ood"
    }
    train_policy_families = {
        str(values_by_id[record_id]["policy_family"])
        for record_id, split in assignments.items()
        if split == "train"
    }
    policy_ood_families_in_train = sorted(policy_ood_families.intersection(train_policy_families))
    held_domain_in_train = sorted(
        record_id
        for record_id, split in assignments.items()
        if split == "train"
        and held_out_domain is not None
        and _extension(records_by_id[record_id])["blueprint"].get("domain") == held_out_domain
    )
    held_language_in_train = sorted(
        record_id
        for record_id, split in assignments.items()
        if split == "train"
        and held_out_language is not None
        and _extension(records_by_id[record_id])["blueprint"].get("language") == held_out_language
    )
    constraints = {
        "no_transitive_group_leakage": not any(leakage_by_field.values()),
        "no_counterfactual_group_leakage": not leakage_by_field["counterfactual_group_id"],
        "no_semantic_duplicate_leakage": not leakage_by_field["semantic_duplicate_cluster"],
        "held_out_domain_absent_from_train": not held_domain_in_train,
        "held_out_language_absent_from_train": not held_language_in_train,
        "requested_splits_populated": not missing_requested,
        "requested_splits_have_independent_components": not underpowered_splits,
        "allocated_targets_within_one_component": not excessive_target_deviation,
        "policy_family_ood_absent_from_train": not policy_ood_families_in_train,
    }
    constraints["all_satisfied"] = all(constraints.values())
    report = {
        "schema_version": "0.1",
        "algorithm": {
            "name": "transitive-metadata-closure-v1",
            "seed": seed,
            "exclusive_partition": True,
            "priority": [
                "test_language_ood",
                "test_domain_ood",
                "test_counterfactual",
                "test_policy_family_ood",
                "human_review_candidates",
                "validation",
                "train",
            ],
            "group_fields": list(_GROUP_FIELDS),
            "language_stratified_group_fields": sorted(_LANGUAGE_STRATIFIED_FIELDS),
            "language_stratification_rationale": (
                "Bilingual realizations of the same authored family are namespaced only for "
                "linkage so a genuine held-out-language view can coexist with other splits; "
                "raw-ID cross-split reuse is reported separately."
            ),
            "globally_atomic_group_fields": [
                "counterfactual_group_id",
                "semantic_duplicate_cluster",
            ],
        },
        "configured_ratios": dict(split_ratios),
        "records": len(records_by_id),
        "active_records": len(active_ids),
        "records_by_split": {split: counts.get(split, 0) for split in ALL_SPLITS},
        "components_by_split": {
            split: component_counts.get(split, 0) for split in PUBLISHED_SPLITS
        },
        "achieved_share_by_split": {
            split: round(counts.get(split, 0) / len(active_ids), 6) for split in PUBLISHED_SPLITS
        },
        "requested_splits": sorted(requested),
        "held_out_domain": held_out_domain,
        "held_out_language": held_out_language,
        "human_review_records_requested": human_review_records,
        "component_diagnostics": {
            **collapse_diagnostics,
            "collapse_warning": largest_share > 0.25,
            "linkages_by_field": {field: linkage_counts.get(field, 0) for field in _GROUP_FIELDS},
            "component_sizes": dict(
                sorted(Counter(component.size for component in components).items())
            ),
            "minimum_components_per_requested_split": effective_minimum_components,
            "underpowered_requested_splits": underpowered_splits,
        },
        "target_deviations": target_deviations,
        "leakage_values_by_field": leakage_by_field,
        "raw_leakage_values_by_field": raw_leakage_by_field,
        "test_policy_family_ood_families": sorted(policy_ood_families),
        "test_policy_family_ood_families_in_train": policy_ood_families_in_train,
        "held_out_domain_train_record_ids": held_domain_in_train,
        "held_out_language_train_record_ids": held_language_in_train,
        "constraints": constraints,
        "assignments": assignments,
    }
    return PolicyBenchSplitResult(
        assignments=assignments,
        records=tuple(updated_records),
        records_by_split={
            split: tuple(sorted(items, key=lambda item: item["id"]))
            for split, items in by_split.items()
        },
        report=report,
    )


def write_policybench_splits(
    result: PolicyBenchSplitResult,
    output: str | Path,
    *,
    include_excluded: bool = False,
) -> dict[str, Any]:
    """Write deterministic split JSONL files, a report, and named checksums."""

    if not isinstance(result, PolicyBenchSplitResult):
        raise PolicyBenchSplitError("result must be a PolicyBenchSplitResult")
    root = Path(output)
    names: list[str] = []
    for split in PUBLISHED_SPLITS:
        name = f"{split}.jsonl"
        write_jsonl(root / name, result.records_by_split[split])
        names.append(name)
    if include_excluded:
        write_jsonl(root / "excluded.jsonl", result.records_by_split["EXCLUDED"])
        names.append("excluded.jsonl")
    write_json(root / "split_report.json", result.report)
    names.append("split_report.json")
    write_named_checksums(root, names)
    return {
        "schema_version": "0.1",
        "output": root.as_posix(),
        "files": sorted(names),
        "checksums": "checksums.sha256",
        "records_by_split": result.report["records_by_split"],
    }


__all__ = [
    "ALL_SPLITS",
    "PUBLISHED_SPLITS",
    "PolicyBenchSplitError",
    "PolicyBenchSplitResult",
    "assign_policybench_splits",
    "write_policybench_splits",
]
