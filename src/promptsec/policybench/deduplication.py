"""Deterministic local duplicate analysis for PromptSec-PolicyBench.

Exact and normalized duplicate decisions are intentionally separate from lexical-
semantic clustering.  Counterfactual siblings form one indivisible duplicate unit:
the pipeline either keeps the complete unit or excludes it as a whole.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from promptsec.data.hashing import sha256_text
from promptsec.data.quality.deduplication import (
    DedupConfig,
    DeduplicationError,
    analyze_duplicates,
    normalize_text,
)

_DECISIONS = (
    "KEEP",
    "KEEP_COUNTERFACTUAL_SIBLING",
    "REJECT_EXACT_DUPLICATE",
    "REJECT_NORMALIZED_DUPLICATE",
    "REJECT_SEMANTIC_DUPLICATE",
)


class PolicyBenchDeduplicationError(ValueError):
    """Raised when PolicyBench records cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class DuplicateAnalysis:
    """Per-record duplicate assignments and a text-free audit report."""

    assignments: dict[str, dict[str, Any]]
    kept_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    report: dict[str, Any]


def _policybench(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    extension = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(extension, Mapping):
        raise PolicyBenchDeduplicationError(
            f"record {record.get('id')!r}: extensions.policybench_v0_1 must be an object"
        )
    for field in ("policy", "blueprint", "grouping"):
        if not isinstance(extension.get(field), Mapping):
            raise PolicyBenchDeduplicationError(
                f"record {record.get('id')!r}: policybench extension {field} must be an object"
            )
    counterfactual = extension.get("counterfactual")
    if counterfactual is not None and not isinstance(counterfactual, Mapping):
        raise PolicyBenchDeduplicationError(
            f"record {record.get('id')!r}: counterfactual must be an object or null"
        )
    return extension


def _counterfactual_group(extension: Mapping[str, Any]) -> str | None:
    counterfactual = extension.get("counterfactual")
    if not isinstance(counterfactual, Mapping):
        return None
    value = counterfactual.get("counterfactual_group_id")
    if not isinstance(value, str) or not value:
        raise PolicyBenchDeduplicationError(
            "counterfactual.counterfactual_group_id must be a non-empty string"
        )
    return value


def _group_report(prefix: str, groups: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "group_id": f"{prefix}_{digest}",
            "member_ids": sorted(member_ids),
            "representative_id": min(member_ids),
            "size": len(member_ids),
        }
        for digest, member_ids in sorted(groups.items())
        if len(member_ids) > 1
    ]


def _selected_duplicate_unit(
    member_ids: list[str],
    counterfactual_by_id: Mapping[str, str | None],
) -> set[str]:
    units: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(member_ids):
        group_id = counterfactual_by_id[record_id]
        unit_id = f"counterfactual:{group_id}" if group_id else f"record:{record_id}"
        units[unit_id].append(record_id)
    selected_key = min(units, key=lambda key: (min(units[key]), key))
    return set(units[selected_key])


def analyze_policybench_duplicates(
    records: Iterable[Mapping[str, Any]],
    *,
    semantic_threshold: float = 0.93,
    reject_exact: bool = True,
    reject_normalized: bool = True,
) -> DuplicateAnalysis:
    """Analyze exact, normalized, and lexical-semantic duplicates locally.

    The lexical-semantic stage reuses PromptSec's deterministic token, bigram, and
    character-trigram Jaccard implementation.  It never loads an embedding model or
    accesses the network.  Input order cannot affect assignments or reports.
    """

    if not isinstance(reject_exact, bool) or not isinstance(reject_normalized, bool):
        raise PolicyBenchDeduplicationError("duplicate rejection flags must be booleans")
    materialized: dict[str, Mapping[str, Any]] = {}
    counterfactual_by_id: dict[str, str | None] = {}
    raw_groups: dict[str, list[str]] = defaultdict(list)
    normalized_groups: dict[str, list[str]] = defaultdict(list)

    for record in records:
        if not isinstance(record, Mapping):
            raise PolicyBenchDeduplicationError("every record must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise PolicyBenchDeduplicationError("every record must have a non-empty string id")
        if record_id in materialized:
            raise PolicyBenchDeduplicationError(f"duplicate canonical record id: {record_id}")
        content = record.get("content")
        text = content.get("text") if isinstance(content, Mapping) else None
        if not isinstance(text, str):
            raise PolicyBenchDeduplicationError(
                f"record {record_id!r}: content.text must be a string"
            )
        extension = _policybench(record)
        materialized[record_id] = record
        counterfactual_by_id[record_id] = _counterfactual_group(extension)
        raw_groups[sha256_text(text)].append(record_id)
        normalized_groups[sha256_text(normalize_text(text))].append(record_id)

    try:
        semantic = analyze_duplicates(
            (materialized[record_id] for record_id in sorted(materialized)),
            DedupConfig(
                semantic_threshold=semantic_threshold,
                variant_threshold=semantic_threshold,
            ),
        )
    except (DeduplicationError, TypeError, ValueError) as error:
        raise PolicyBenchDeduplicationError(str(error)) from error

    decisions = {record_id: "KEEP" for record_id in materialized}
    duplicate_groups = normalized_groups if reject_normalized else raw_groups
    if reject_exact or reject_normalized:
        for _digest, member_ids in sorted(duplicate_groups.items()):
            if len(member_ids) < 2:
                continue
            selected = _selected_duplicate_unit(member_ids, counterfactual_by_id)
            representative_id = min(selected)
            representative_raw_hash = semantic.assignments[representative_id]["raw_hash"]
            for record_id in sorted(member_ids):
                if record_id in selected:
                    if len(selected) > 1 and counterfactual_by_id[record_id] is not None:
                        decisions[record_id] = "KEEP_COUNTERFACTUAL_SIBLING"
                    continue
                same_raw = semantic.assignments[record_id]["raw_hash"] == representative_raw_hash
                if same_raw and reject_exact:
                    decisions[record_id] = "REJECT_EXACT_DUPLICATE"
                elif reject_normalized:
                    decisions[record_id] = "REJECT_NORMALIZED_DUPLICATE"

    # A counterfactual family is scientifically indivisible. If any member loses
    # duplicate-unit selection, exclude every sibling rather than retaining a
    # partial family that could no longer support the declared comparison.
    counterfactual_members: dict[str, list[str]] = defaultdict(list)
    for record_id, group_id in counterfactual_by_id.items():
        if group_id is not None:
            counterfactual_members[group_id].append(record_id)
    for _group_id, member_ids in sorted(counterfactual_members.items()):
        rejection_decisions = sorted(
            {
                decisions[record_id]
                for record_id in member_ids
                if decisions[record_id].startswith("REJECT_")
            }
        )
        if rejection_decisions:
            propagated = (
                "REJECT_NORMALIZED_DUPLICATE"
                if "REJECT_NORMALIZED_DUPLICATE" in rejection_decisions
                else "REJECT_EXACT_DUPLICATE"
            )
            for record_id in member_ids:
                decisions[record_id] = propagated

    semantic_members: dict[str, list[str]] = defaultdict(list)
    for record_id, assignment in semantic.assignments.items():
        semantic_members[assignment["semantic_cluster_id"]].append(record_id)
    for _cluster_id, member_ids in sorted(semantic_members.items()):
        if len(member_ids) < 2:
            continue
        selected = _selected_duplicate_unit(member_ids, counterfactual_by_id)
        for record_id in sorted(member_ids):
            if record_id not in selected and not decisions[record_id].startswith("REJECT_"):
                decisions[record_id] = "REJECT_SEMANTIC_DUPLICATE"
            elif (
                record_id in selected
                and len(selected) > 1
                and counterfactual_by_id[record_id] is not None
                and decisions[record_id] == "KEEP"
            ):
                decisions[record_id] = "KEEP_COUNTERFACTUAL_SIBLING"

    # Semantic rejection is just as binding as byte/normalized rejection. Propagate
    # it to the complete counterfactual unit so no scientifically partial pair can
    # survive corpus validation.
    rejection_priority = (
        "REJECT_NORMALIZED_DUPLICATE",
        "REJECT_EXACT_DUPLICATE",
        "REJECT_SEMANTIC_DUPLICATE",
    )
    for _group_id, member_ids in sorted(counterfactual_members.items()):
        propagated = next(
            (
                decision
                for decision in rejection_priority
                if any(decisions[record_id] == decision for record_id in member_ids)
            ),
            None,
        )
        if propagated is not None:
            for record_id in member_ids:
                decisions[record_id] = propagated

    assignments: dict[str, dict[str, Any]] = {}
    for record_id in sorted(materialized):
        source = semantic.assignments[record_id]
        raw_hash = source["raw_hash"]
        normalized_hash = source["normalized_hash"]
        assignments[record_id] = {
            "raw_hash": raw_hash,
            "normalized_hash": normalized_hash,
            "contextual_hash": source["contextual_hash"],
            "exact_duplicate_group_id": f"exact_{raw_hash}",
            "normalized_duplicate_group_id": f"normalized_{normalized_hash}",
            "semantic_duplicate_cluster_id": source["semantic_cluster_id"],
            "semantic_representative_id": source["representative_id"],
            "similarity_to_representative": source["similarity_to_representative"],
            "decision": decisions[record_id],
        }

    rejected = tuple(
        record_id
        for record_id in sorted(assignments)
        if assignments[record_id]["decision"].startswith("REJECT_")
    )
    kept = tuple(record_id for record_id in sorted(assignments) if record_id not in rejected)
    exact_report = _group_report("exact", raw_groups)
    normalized_report = _group_report("normalized", normalized_groups)
    semantic_report = [
        {
            "cluster_id": cluster_id,
            "member_ids": sorted(member_ids),
            "representative_id": min(
                member_ids,
                key=lambda record_id: (
                    semantic.assignments[record_id]["representative_id"] != record_id,
                    record_id,
                ),
            ),
            "size": len(member_ids),
        }
        for cluster_id, member_ids in sorted(semantic_members.items())
    ]
    decision_counts = Counter(decisions.values())
    report = {
        "schema_version": "0.1",
        "algorithm": {
            "name": "promptsec-local-lexical-semantic",
            "network_model_required": False,
            "normalization": "NFKC + Unicode casefold + collapsed whitespace",
            "semantic_features": [
                "token_jaccard",
                "token_bigram_jaccard",
                "character_trigram_jaccard",
            ],
            "semantic_threshold": float(semantic_threshold),
        },
        "policy": {
            "reject_exact": reject_exact,
            "reject_normalized": reject_normalized,
            "reject_semantic_at_or_above_threshold": True,
            "counterfactual_units_are_atomic": True,
        },
        "summary": {
            "records": len(materialized),
            "kept_records": len(kept),
            "rejected_records": len(rejected),
            "exact_duplicate_groups": len(exact_report),
            "normalized_duplicate_groups": len(normalized_report),
            "semantic_duplicate_clusters": sum(cluster["size"] > 1 for cluster in semantic_report),
        },
        "decision_counts": {decision: decision_counts.get(decision, 0) for decision in _DECISIONS},
        "exact_duplicate_groups": exact_report,
        "normalized_duplicate_groups": normalized_report,
        "semantic_clusters": semantic_report,
    }
    return DuplicateAnalysis(
        assignments=assignments,
        kept_ids=kept,
        rejected_ids=rejected,
        report=report,
    )


__all__ = [
    "DuplicateAnalysis",
    "PolicyBenchDeduplicationError",
    "analyze_policybench_duplicates",
    "normalize_text",
]
