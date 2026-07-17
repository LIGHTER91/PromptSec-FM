"""Deterministic, manifest-bound stratified selections from a PolicyBench plan."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import canonical_json_bytes, sha256_file, sha256_text
from promptsec.policybench.blueprints import BlueprintPlan, ScenarioBlueprint
from promptsec.policybench.config import CATEGORY_ORDER, COUNTERFACTUAL_TYPE_ORDER
from promptsec.policybench.io import read_json_object

_MANIFEST_KEYS = {
    "schema_version",
    "manifest_type",
    "selector",
    "seed",
    "source_config_sha256",
    "source_plan_sha256",
    "source_plan_records",
    "records_per_category",
    "selected_records",
    "selected_plan_sha256",
    "selected_scenario_ids",
    "category_counts",
    "domain_counts",
    "language_counts",
    "policy_ids",
    "policy_id_counts",
    "counterfactual_group_ids",
    "counterfactual_type_counts",
    "source_role_counts",
    "authority_status_expectations",
    "policy_alignment_expectations",
    "attack_family_expectations",
    "selection_records",
}
_MAXIMUM_MANIFEST_BYTES = 5_000_000


class PolicyBenchSelectionError(ValueError):
    """Raised when a selection cannot preserve its declared scientific constraints."""


@dataclass(frozen=True, slots=True)
class _Unit:
    unit_id: str
    members: tuple[ScenarioBlueprint, ...]


def plan_sha256(blueprints: Iterable[ScenarioBlueprint]) -> str:
    """Hash a blueprint collection using the canonical release-plan representation."""

    return hashlib.sha256(
        b"\n".join(
            canonical_json_bytes(item.to_dict())
            for item in sorted(blueprints, key=lambda value: value.scenario_id)
        )
    ).hexdigest()


def _units(blueprints: Iterable[ScenarioBlueprint]) -> tuple[_Unit, ...]:
    grouped: dict[str, list[ScenarioBlueprint]] = defaultdict(list)
    for blueprint in blueprints:
        provenance = blueprint.counterfactual
        unit_id = (
            f"counterfactual:{provenance.counterfactual_group_id}"
            if provenance is not None
            else f"scenario:{blueprint.scenario_id}"
        )
        grouped[unit_id].append(blueprint)
    result: list[_Unit] = []
    for unit_id, members in grouped.items():
        parent_id = (
            members[0].counterfactual.parent_scenario_id if members[0].counterfactual else None
        )
        members.sort(key=lambda item: (item.scenario_id != parent_id, item.scenario_id))
        if parent_id is not None and (len(members) != 2 or members[0].scenario_id != parent_id):
            raise PolicyBenchSelectionError(f"partial counterfactual unit in plan: {unit_id}")
        result.append(_Unit(unit_id=unit_id, members=tuple(members)))
    return tuple(sorted(result, key=lambda item: item.unit_id))


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _annotations(blueprint: ScenarioBlueprint) -> tuple[str, str, tuple[str, ...]]:
    expected = blueprint.expected_annotations
    return (
        expected.authority_status,
        expected.protected_policy_alignment,
        expected.attack_families,
    )


def _stable_key(seed: int, context: str, unit_id: str) -> tuple[str, str]:
    return sha256_text(f"{seed}:stratified-pilot-v1:{context}:{unit_id}"), unit_id


def select_stratified_blueprints(
    plan: BlueprintPlan,
    *,
    seed: int,
    records_per_category: int,
) -> tuple[ScenarioBlueprint, ...]:
    """Select exact category quotas while keeping every chosen counterfactual pair whole.

    One complete group of each configured counterfactual type is chosen first. The
    remaining cells are filled from non-counterfactual units with deterministic
    coverage-aware tie breaking. The source plan itself is never mutated.
    """

    if isinstance(records_per_category, bool) or not isinstance(records_per_category, int):
        raise PolicyBenchSelectionError("records_per_category must be an integer")
    if records_per_category < 1:
        raise PolicyBenchSelectionError("records_per_category must be positive")
    units = _units(plan.blueprints)
    selected: list[_Unit] = []
    selected_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    policies: set[str] = set()
    source_roles: set[str] = set()
    authorities: set[str] = set()
    alignments: set[str] = set()
    attack_families: set[str] = set()

    def add(unit: _Unit) -> None:
        selected.append(unit)
        selected_ids.add(unit.unit_id)
        for member in unit.members:
            category_counts[member.category] += 1
            domain_counts[member.domain] += 1
            language_counts[member.language] += 1
            policies.add(member.policy_id)
            source_roles.add(member.source_role)
            authority, alignment, families = _annotations(member)
            authorities.add(authority)
            alignments.add(alignment)
            attack_families.update(families)

    def fits(unit: _Unit) -> bool:
        additions = Counter(member.category for member in unit.members)
        return all(
            category_counts[category] + count <= records_per_category
            for category, count in additions.items()
        )

    # Preserve broad counterfactual scientific coverage without allowing pairs to
    # crowd out an exact category marginal.
    for counterfactual_type in COUNTERFACTUAL_TYPE_ORDER:
        candidates = [
            unit
            for unit in units
            if len(unit.members) == 2
            and unit.members[0].counterfactual is not None
            and unit.members[0].counterfactual.counterfactual_type == counterfactual_type
            and fits(unit)
        ]
        if not candidates:
            raise PolicyBenchSelectionError(
                f"cannot select a complete {counterfactual_type} group within category quotas"
            )

        def group_key(
            unit: _Unit, counterfactual_type: str = counterfactual_type
        ) -> tuple[Any, ...]:
            member_authorities = {
                item.expected_annotations.authority_status for item in unit.members
            }
            member_alignments = {
                item.expected_annotations.protected_policy_alignment for item in unit.members
            }
            member_families = {
                family
                for item in unit.members
                for family in item.expected_annotations.attack_families
            }
            return (
                -len({item.category for item in unit.members}.difference(category_counts)),
                -len({item.domain for item in unit.members}.difference(domain_counts)),
                -len({item.language for item in unit.members}.difference(language_counts)),
                -len(member_authorities.difference(authorities)),
                -len(member_alignments.difference(alignments)),
                -len(member_families.difference(attack_families)),
                -len({item.source_role for item in unit.members}.difference(source_roles)),
                sum(category_counts[item.category] for item in unit.members),
                _stable_key(seed, counterfactual_type, unit.unit_id),
            )

        add(min(candidates, key=group_key))

    singleton_by_category: dict[str, list[_Unit]] = defaultdict(list)
    for unit in units:
        if len(unit.members) == 1:
            singleton_by_category[unit.members[0].category].append(unit)

    # Fill every category independently. New language/domain cells within the
    # category are preferred, then global expectation and policy diversity.
    for category in CATEGORY_ORDER:
        category_domains = {
            member.domain
            for unit in selected
            for member in unit.members
            if member.category == category
        }
        category_languages = {
            member.language
            for unit in selected
            for member in unit.members
            if member.category == category
        }
        category_domain_counts = Counter(
            member.domain
            for unit in selected
            for member in unit.members
            if member.category == category
        )
        category_language_counts = Counter(
            member.language
            for unit in selected
            for member in unit.members
            if member.category == category
        )
        while category_counts[category] < records_per_category:
            candidates = [
                unit for unit in singleton_by_category[category] if unit.unit_id not in selected_ids
            ]
            if not candidates:
                raise PolicyBenchSelectionError(
                    f"not enough non-counterfactual {category} records to reach exact quota"
                )

            def singleton_key(
                unit: _Unit,
                category: str = category,
                category_domains: set[str] = category_domains,
                category_languages: set[str] = category_languages,
                category_domain_counts: Counter[str] = category_domain_counts,
                category_language_counts: Counter[str] = category_language_counts,
            ) -> tuple[Any, ...]:
                member = unit.members[0]
                authority, alignment, families = _annotations(member)
                return (
                    member.language in category_languages,
                    member.domain in category_domains,
                    authority in authorities,
                    alignment in alignments,
                    all(family in attack_families for family in families),
                    member.source_role in source_roles,
                    member.policy_id in policies,
                    category_language_counts[member.language],
                    category_domain_counts[member.domain],
                    language_counts[member.language],
                    domain_counts[member.domain],
                    _stable_key(seed, category, unit.unit_id),
                )

            chosen = min(candidates, key=singleton_key)
            member = chosen.members[0]
            add(chosen)
            category_domains.add(member.domain)
            category_languages.add(member.language)
            category_domain_counts[member.domain] += 1
            category_language_counts[member.language] += 1

    result = tuple(
        sorted(
            (member for unit in selected for member in unit.members),
            key=lambda item: item.scenario_id,
        )
    )
    expected_total = records_per_category * len(CATEGORY_ORDER)
    if (
        len(result) != expected_total
        or len({item.scenario_id for item in result}) != expected_total
    ):
        raise PolicyBenchSelectionError("stratified selection did not produce unique exact total")
    if Counter(item.category for item in result) != Counter(
        {category: records_per_category for category in CATEGORY_ORDER}
    ):
        raise PolicyBenchSelectionError("stratified selection did not produce exact categories")
    _require_complete_units(plan.blueprints, result)
    return result


def _require_complete_units(
    source_blueprints: Iterable[ScenarioBlueprint], selected: Iterable[ScenarioBlueprint]
) -> None:
    selected_ids = {item.scenario_id for item in selected}
    for unit in _units(source_blueprints):
        member_ids = {item.scenario_id for item in unit.members}
        intersection = selected_ids.intersection(member_ids)
        if intersection and intersection != member_ids:
            raise PolicyBenchSelectionError(f"selection splits {unit.unit_id}")


def build_selection_manifest(
    plan: BlueprintPlan,
    selected: Sequence[ScenarioBlueprint],
    *,
    seed: int,
    records_per_category: int,
    source_config_sha256: str,
) -> dict[str, Any]:
    """Describe a selection completely without including generated candidate text."""

    selected_values = tuple(sorted(selected, key=lambda item: item.scenario_id))
    _require_complete_units(plan.blueprints, selected_values)
    if len({item.scenario_id for item in selected_values}) != len(selected_values):
        raise PolicyBenchSelectionError("selection contains duplicate scenario IDs")
    category_counts = Counter(item.category for item in selected_values)
    expected_categories = Counter({category: records_per_category for category in CATEGORY_ORDER})
    if category_counts != expected_categories:
        raise PolicyBenchSelectionError("selection manifest requires exact category quotas")
    counterfactual_group_ids = sorted(
        {
            item.counterfactual.counterfactual_group_id
            for item in selected_values
            if item.counterfactual is not None
        }
    )
    return {
        "schema_version": "0.1",
        "manifest_type": "policybench_stratified_selection",
        "selector": "deterministic_stratified_v1",
        "seed": seed,
        "source_config_sha256": source_config_sha256,
        "source_plan_sha256": plan_sha256(plan.blueprints),
        "source_plan_records": len(plan.blueprints),
        "records_per_category": records_per_category,
        "selected_records": len(selected_values),
        "selected_plan_sha256": plan_sha256(selected_values),
        "selected_scenario_ids": [item.scenario_id for item in selected_values],
        "category_counts": _counter(item.category for item in selected_values),
        "domain_counts": _counter(item.domain for item in selected_values),
        "language_counts": _counter(item.language for item in selected_values),
        "policy_ids": sorted({item.policy_id for item in selected_values}),
        "policy_id_counts": _counter(item.policy_id for item in selected_values),
        "counterfactual_group_ids": counterfactual_group_ids,
        "counterfactual_type_counts": _counter(
            item.counterfactual.counterfactual_type
            for item in selected_values
            if item.counterfactual is not None
        ),
        "source_role_counts": _counter(item.source_role for item in selected_values),
        "authority_status_expectations": _counter(
            item.expected_annotations.authority_status for item in selected_values
        ),
        "policy_alignment_expectations": _counter(
            item.expected_annotations.protected_policy_alignment for item in selected_values
        ),
        "attack_family_expectations": _counter(
            family
            for item in selected_values
            for family in item.expected_annotations.attack_families
        ),
        "selection_records": [
            {
                "scenario_id": item.scenario_id,
                "blueprint_sha256": item.sha256(),
                "category": item.category,
                "domain": item.domain,
                "language": item.language,
                "policy_id": item.policy_id,
                "counterfactual_group_id": (
                    item.counterfactual.counterfactual_group_id if item.counterfactual else None
                ),
                "counterfactual_type": (
                    item.counterfactual.counterfactual_type if item.counterfactual else None
                ),
                "source_role": item.source_role,
                "authority_status_expectation": item.expected_annotations.authority_status,
                "policy_alignment_expectation": (
                    item.expected_annotations.protected_policy_alignment
                ),
                "attack_family_expectations": list(item.expected_annotations.attack_families),
            }
            for item in selected_values
        ],
    }


def load_selection_manifest(
    path: str | Path,
    plan: BlueprintPlan,
    *,
    seed: int,
    source_config_sha256: str,
) -> tuple[tuple[ScenarioBlueprint, ...], str]:
    """Strictly load and bind a manifest to its exact source and selected plans."""

    manifest_path = Path(path).resolve()
    value = read_json_object(manifest_path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
    if set(value) != _MANIFEST_KEYS:
        raise PolicyBenchSelectionError("selection manifest has unexpected or missing fields")
    if value.get("schema_version") != "0.1" or value.get("manifest_type") != (
        "policybench_stratified_selection"
    ):
        raise PolicyBenchSelectionError("unsupported selection manifest contract")
    if value.get("selector") != "deterministic_stratified_v1":
        raise PolicyBenchSelectionError("unsupported selection algorithm")
    if value.get("seed") != seed:
        raise PolicyBenchSelectionError("selection manifest seed does not match generation")
    if value.get("source_config_sha256") != source_config_sha256:
        raise PolicyBenchSelectionError("selection manifest config hash does not match")
    if value.get("source_plan_sha256") != plan_sha256(plan.blueprints):
        raise PolicyBenchSelectionError("selection manifest source-plan hash does not match")
    scenario_ids = value.get("selected_scenario_ids")
    if not isinstance(scenario_ids, list) or not all(
        isinstance(item, str) and item for item in scenario_ids
    ):
        raise PolicyBenchSelectionError("selected_scenario_ids must be an array of strings")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise PolicyBenchSelectionError("selection manifest contains duplicate scenario IDs")
    by_id = {item.scenario_id: item for item in plan.blueprints}
    unexpected = sorted(set(scenario_ids).difference(by_id))
    if unexpected:
        raise PolicyBenchSelectionError(
            f"selection manifest contains unexpected scenario IDs: {unexpected[:3]}"
        )
    selected = tuple(
        sorted((by_id[item] for item in scenario_ids), key=lambda item: item.scenario_id)
    )
    records_per_category = value.get("records_per_category")
    if isinstance(records_per_category, bool) or not isinstance(records_per_category, int):
        raise PolicyBenchSelectionError("records_per_category must be an integer")
    expected = build_selection_manifest(
        plan,
        selected,
        seed=seed,
        records_per_category=records_per_category,
        source_config_sha256=source_config_sha256,
    )
    if value != expected:
        raise PolicyBenchSelectionError(
            "selection manifest metadata does not match its selected blueprints"
        )
    return selected, sha256_file(manifest_path)


__all__ = [
    "PolicyBenchSelectionError",
    "build_selection_manifest",
    "load_selection_manifest",
    "plan_sha256",
    "select_stratified_blueprints",
]
