"""Deterministic, cluster-atomic dataset splitting with explicit holdouts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

_RATIO_SPLITS = ("train", "validation", "test_id")
_OUTPUT_SPLITS = (
    "train",
    "validation",
    "test_id",
    "test_held_out_source",
    "test_held_out_family",
)
_DEDUP_DECISIONS = {
    "KEEP",
    "DROP_EXACT_DUPLICATE",
    "KEEP_VARIANT",
    "REVIEW",
}


class SplitError(ValueError):
    """Raised when split inputs cannot satisfy the audited contract."""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuration for deterministic cluster assignment."""

    seed: int = 0
    held_out_source: str | None = None
    held_out_family: str | None = None
    general_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test_id": 0.1}
    )
    notinject_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.2, "validation": 0.1, "test_id": 0.7}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SplitError("seed must be an integer")
        for name in ("held_out_source", "held_out_family"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise SplitError(f"{name} must be a non-empty string or None")
        object.__setattr__(
            self,
            "general_ratios",
            _validated_ratios(self.general_ratios, "general_ratios"),
        )
        object.__setattr__(
            self,
            "notinject_ratios",
            _validated_ratios(self.notinject_ratios, "notinject_ratios"),
        )


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Per-record assignments, materialized splits, and audit report."""

    assignments: dict[str, str]
    splits: dict[str, list[str]]
    report: dict[str, Any]


def assign_splits(
    records: Iterable[Mapping[str, Any]],
    dedup_assignments: Any,
    family_assignments: Any,
    config: SplitConfig,
) -> SplitResult:
    """Assign records atomically by semantic cluster.

    Exact duplicates are marked ``DROPPED_EXACT`` and never materialized.
    Holdout-source clusters take precedence over holdout-family clusters; all
    remaining clusters use a stable SHA-256 draw and the applicable ratio set.
    """

    if not isinstance(config, SplitConfig):
        raise SplitError("config must be a SplitConfig")
    # Revalidate copied mappings in case a caller retained and mutated them.
    general_ratios = _validated_ratios(config.general_ratios, "general_ratios")
    notinject_ratios = _validated_ratios(config.notinject_ratios, "notinject_ratios")

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise SplitError("every record must be a mapping")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise SplitError("every record must have a non-empty string id")
        if record_id in records_by_id:
            raise SplitError(f"duplicate record id: {record_id}")
        records_by_id[record_id] = record

    dedup = _assignment_mapping(dedup_assignments, "dedup_assignments")
    families = _assignment_mapping(family_assignments, "family_assignments")
    _require_assignments(records_by_id, dedup, "dedup_assignments")
    _require_assignments(records_by_id, families, "family_assignments")

    record_clusters: dict[str, str] = {}
    dedup_decisions: dict[str, str] = {}
    family_by_id: dict[str, str] = {}
    domain_by_id: dict[str, str] = {}
    source_by_id: dict[str, str] = {}
    cluster_members: defaultdict[str, list[str]] = defaultdict(list)

    for record_id in sorted(records_by_id):
        dedup_entry = dedup[record_id]
        decision = _dedup_decision(dedup_entry)
        cluster_id = _semantic_cluster_id(record_id, dedup_entry)
        family, domain = _family_and_domain(record_id, families[record_id])
        source = _source_id(records_by_id[record_id])

        dedup_decisions[record_id] = decision
        record_clusters[record_id] = cluster_id
        family_by_id[record_id] = family
        domain_by_id[record_id] = domain
        source_by_id[record_id] = source
        cluster_members[cluster_id].append(record_id)

    assignments: dict[str, str] = {}
    splits: dict[str, list[str]] = {name: [] for name in _OUTPUT_SPLITS}
    cluster_assignments: dict[str, str] = {}
    ratio_policy_by_cluster: dict[str, str] = {}

    for record_id, decision in dedup_decisions.items():
        if decision == "DROP_EXACT_DUPLICATE":
            assignments[record_id] = "DROPPED_EXACT"

    cluster_sources_by_id = {
        cluster_id: {source_by_id[record_id] for record_id in member_ids}
        for cluster_id, member_ids in cluster_members.items()
    }
    cluster_families_by_id = {
        cluster_id: {family_by_id[record_id] for record_id in member_ids}
        for cluster_id, member_ids in cluster_members.items()
    }
    held_out_source_clusters = {
        cluster_id
        for cluster_id, sources in cluster_sources_by_id.items()
        if config.held_out_source and config.held_out_source in sources
    }
    held_out_family_clusters, held_out_family_closure = _family_holdout_closure(
        cluster_families_by_id,
        held_out_source_clusters,
        config.held_out_family,
    )

    for cluster_id in sorted(cluster_members):
        all_member_ids = sorted(cluster_members[cluster_id])
        kept_member_ids = [
            record_id
            for record_id in all_member_ids
            if dedup_decisions[record_id] != "DROP_EXACT_DUPLICATE"
        ]
        if not kept_member_ids:
            continue

        cluster_sources = cluster_sources_by_id[cluster_id]
        if cluster_id in held_out_source_clusters:
            split = "test_held_out_source"
            ratio_policy = "held_out_source"
        elif cluster_id in held_out_family_clusters:
            split = "test_held_out_family"
            ratio_policy = "held_out_family"
        else:
            use_notinject_policy = any(
                _normalized_source(source) == "notinject" for source in cluster_sources
            )
            ratios = notinject_ratios if use_notinject_policy else general_ratios
            ratio_policy = "notinject" if use_notinject_policy else "general"
            split = _hashed_split(
                cluster_id,
                seed=config.seed,
                ratios=ratios,
            )

        cluster_assignments[cluster_id] = split
        ratio_policy_by_cluster[cluster_id] = ratio_policy
        for record_id in kept_member_ids:
            assignments[record_id] = split
            splits[split].append(record_id)

    assignments = dict(sorted(assignments.items()))
    for split_ids in splits.values():
        split_ids.sort()

    report = _build_report(
        records_by_id=records_by_id,
        assignments=assignments,
        splits=splits,
        record_clusters=record_clusters,
        cluster_assignments=cluster_assignments,
        ratio_policy_by_cluster=ratio_policy_by_cluster,
        dedup_decisions=dedup_decisions,
        source_by_id=source_by_id,
        family_by_id=family_by_id,
        domain_by_id=domain_by_id,
        config=config,
        general_ratios=general_ratios,
        notinject_ratios=notinject_ratios,
        held_out_family_closure=held_out_family_closure,
    )
    return SplitResult(assignments=assignments, splits=splits, report=report)


def _family_holdout_closure(
    cluster_families: Mapping[str, set[str]],
    held_out_source_clusters: set[str],
    configured_family: str | None,
) -> tuple[set[str], set[str]]:
    """Expand the family holdout until it is disjoint from all other splits."""

    if configured_family is None:
        return set(), set()
    closure = {configured_family}
    selected: set[str] = set()
    changed = True
    while changed:
        changed = False
        for cluster_id in sorted(cluster_families):
            if cluster_id in held_out_source_clusters or cluster_id in selected:
                continue
            families = cluster_families[cluster_id]
            if families.isdisjoint(closure):
                continue
            selected.add(cluster_id)
            before = len(closure)
            closure.update(families)
            changed = changed or len(closure) != before
        # Selecting a cluster without adding a family still completes in this pass;
        # a new family triggers another pass to pull every cluster that shares it.
    return selected, closure


def _build_report(
    *,
    records_by_id: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    splits: Mapping[str, list[str]],
    record_clusters: Mapping[str, str],
    cluster_assignments: Mapping[str, str],
    ratio_policy_by_cluster: Mapping[str, str],
    dedup_decisions: Mapping[str, str],
    source_by_id: Mapping[str, str],
    family_by_id: Mapping[str, str],
    domain_by_id: Mapping[str, str],
    config: SplitConfig,
    general_ratios: Mapping[str, float],
    notinject_ratios: Mapping[str, float],
    held_out_family_closure: set[str],
) -> dict[str, Any]:
    train_ids = set(splits["train"])
    held_out_source_in_train = sorted(
        record_id
        for record_id in train_ids
        if config.held_out_source and source_by_id[record_id] == config.held_out_source
    )
    held_out_family_in_train = sorted(
        record_id
        for record_id in train_ids
        if config.held_out_family and family_by_id[record_id] == config.held_out_family
    )

    cluster_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record_id, split in assignments.items():
        if split != "DROPPED_EXACT":
            cluster_splits[record_clusters[record_id]].add(split)
    cluster_leakage = {
        cluster_id: sorted(assigned_splits)
        for cluster_id, assigned_splits in sorted(cluster_splits.items())
        if len(assigned_splits) > 1
    }

    expected_dropped = {
        record_id
        for record_id, decision in dedup_decisions.items()
        if decision == "DROP_EXACT_DUPLICATE"
    }
    materialized_ids = {record_id for split_ids in splits.values() for record_id in split_ids}
    dropped_exact_ids = sorted(expected_dropped)
    exact_duplicates_excluded = all(
        assignments.get(record_id) == "DROPPED_EXACT" and record_id not in materialized_ids
        for record_id in expected_dropped
    )
    no_held_out_source_in_train = not held_out_source_in_train
    no_held_out_family_in_train = not held_out_family_in_train
    train_families = {family_by_id[record_id] for record_id in train_ids}
    test_held_out_families = {
        family_by_id[record_id] for record_id in splits["test_held_out_family"]
    }
    held_out_family_train_overlap = sorted(train_families & test_held_out_families)
    no_held_out_family_train_overlap = not held_out_family_train_overlap
    no_cluster_leakage = not cluster_leakage
    all_satisfied = (
        no_held_out_source_in_train
        and no_held_out_family_in_train
        and no_held_out_family_train_overlap
        and no_cluster_leakage
        and exact_duplicates_excluded
    )

    records_by_split = {name: len(splits[name]) for name in _OUTPUT_SPLITS}
    records_by_split["DROPPED_EXACT"] = len(dropped_exact_ids)
    clusters_by_split = dict(sorted(Counter(cluster_assignments.values()).items()))
    sources_by_split = _distribution_by_split(splits, source_by_id)
    families_by_split = _distribution_by_split(splits, family_by_id)
    domains_by_split = _distribution_by_split(splits, domain_by_id)

    return {
        "seed": config.seed,
        "total_records": len(records_by_id),
        "kept_records": len(records_by_id) - len(dropped_exact_ids),
        "dropped_exact_ids": dropped_exact_ids,
        "policy": {
            "priority": ["held_out_source", "held_out_family", "stable_hash"],
            "held_out_source": config.held_out_source,
            "held_out_family": config.held_out_family,
            "held_out_family_closure": sorted(held_out_family_closure),
            "general_ratios": dict(general_ratios),
            "notinject_ratios": dict(notinject_ratios),
            "hash": "sha256(seed, semantic_cluster_id) with ratio thresholds",
            "ratio_policy_by_cluster": dict(sorted(ratio_policy_by_cluster.items())),
        },
        "constraints": {
            "no_held_out_source_in_train": no_held_out_source_in_train,
            "no_held_out_family_in_train": no_held_out_family_in_train,
            "no_template_family_overlap_with_train": no_held_out_family_train_overlap,
            "no_cluster_leakage": no_cluster_leakage,
            "exact_duplicates_excluded": exact_duplicates_excluded,
            "all_satisfied": all_satisfied,
            "held_out_source_in_train": held_out_source_in_train,
            "held_out_family_in_train": held_out_family_in_train,
            "template_family_overlap_with_train": held_out_family_train_overlap,
            "cluster_leakage": cluster_leakage,
        },
        "distributions": {
            "records_by_split": records_by_split,
            "clusters_by_split": clusters_by_split,
            "sources_by_split": sources_by_split,
            "families_by_split": families_by_split,
            "domains_by_split": domains_by_split,
            "dedup_decisions": dict(sorted(Counter(dedup_decisions.values()).items())),
        },
    }


def _distribution_by_split(
    splits: Mapping[str, list[str]],
    values_by_id: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(values_by_id[record_id] for record_id in ids).items()))
        for split, ids in splits.items()
    }


def _assignment_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    nested = getattr(value, "assignments", None)
    if isinstance(nested, Mapping):
        return nested
    raise SplitError(f"{name} must be a mapping or expose a mapping .assignments")


def _require_assignments(
    records_by_id: Mapping[str, Any],
    assignments: Mapping[str, Any],
    name: str,
) -> None:
    missing = sorted(set(records_by_id) - set(assignments))
    if missing:
        raise SplitError(f"{name} missing record ids: {missing}")


def _dedup_decision(entry: Any) -> str:
    if isinstance(entry, str) and entry in {"DROP_EXACT_DUPLICATE", "DROPPED_EXACT"}:
        return "DROP_EXACT_DUPLICATE"
    decision = _entry_value(entry, "dedup_decision", "KEEP")
    if decision not in _DEDUP_DECISIONS:
        raise SplitError(f"unsupported dedup_decision: {decision!r}")
    return str(decision)


def _semantic_cluster_id(record_id: str, entry: Any) -> str:
    if isinstance(entry, str):
        if entry in {"DROP_EXACT_DUPLICATE", "DROPPED_EXACT"}:
            return record_id
        return entry or record_id
    for field_name in (
        "semantic_cluster_id",
        "cluster_id",
        "representative_id",
        "exact_group_id",
    ):
        value = _entry_value(entry, field_name, None)
        if value is not None and str(value):
            return str(value)
    return record_id


def _family_and_domain(record_id: str, entry: Any) -> tuple[str, str]:
    if isinstance(entry, str) and entry:
        return entry, "unknown"
    family = _entry_value(entry, "template_family", None)
    domain = _entry_value(entry, "domain", "unknown")
    if not isinstance(family, str) or not family:
        raise SplitError(f"family_assignments[{record_id!r}] has no template_family")
    if not isinstance(domain, str) or not domain:
        domain = "unknown"
    return family, domain


def _entry_value(entry: Any, field_name: str, default: Any) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(field_name, default)
    return getattr(entry, field_name, default)


def _source_id(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    provenance = metadata.get("dataset_provenance") if isinstance(metadata, Mapping) else None
    source_dataset = provenance.get("source_dataset") if isinstance(provenance, Mapping) else None
    source_id = source_dataset.get("id") if isinstance(source_dataset, Mapping) else None
    return source_id if isinstance(source_id, str) and source_id else "unknown"


def _normalized_source(source: str) -> str:
    return source.casefold().replace("-", "_")


def _hashed_split(
    cluster_id: str,
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> str:
    payload = json.dumps(
        {
            "seed": seed,
            "semantic_cluster_id": cluster_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(payload).digest(), "big") / (1 << 256)
    cumulative = 0.0
    for split in _RATIO_SPLITS:
        cumulative += ratios[split]
        if draw < cumulative:
            return split
    return _RATIO_SPLITS[-1]


def _validated_ratios(ratios: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(ratios, Mapping):
        raise SplitError(f"{name} must be a mapping")
    if set(ratios) != set(_RATIO_SPLITS):
        raise SplitError(f"{name} must have exactly these keys: {list(_RATIO_SPLITS)}")
    validated: dict[str, float] = {}
    for split in _RATIO_SPLITS:
        value = ratios[split]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise SplitError(f"{name}.{split} must be a finite non-negative number")
        validated[split] = float(value)
    if not math.isclose(sum(validated.values()), 1.0, abs_tol=1e-9):
        raise SplitError(f"{name} values must sum to 1.0")
    return validated


__all__ = ["SplitConfig", "SplitError", "SplitResult", "assign_splits"]
