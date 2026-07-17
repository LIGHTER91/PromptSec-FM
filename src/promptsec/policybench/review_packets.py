"""Deterministic, blinded double-annotation packets for PolicyBench SILVER records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file, sha256_text
from promptsec.policybench.io import write_json, write_jsonl, write_named_checksums

_ANNOTATION_FIELDS = (
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "attack_families",
    "attack_objectives",
    "spans",
    "annotation_status",
    "annotator_confidence",
)
_CONTENT_FIELDS = (
    "text",
    "language",
    "delivery_mode",
    "source_role",
    "content_origin",
    "ingestion_path",
    "modality",
    "source_integrity",
)


class PolicyBenchReviewError(ValueError):
    """Raised when a blinded packet cannot be constructed without leakage."""


@dataclass(frozen=True, slots=True)
class ReviewPacketResult:
    """Packet paths and machine-readable selection/manifest documents."""

    output: Path
    selected_ids: tuple[str, ...]
    selection_report: dict[str, Any]
    packet_manifest: dict[str, Any]


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(value, Mapping):
        raise PolicyBenchReviewError(
            f"record {record.get('id')!r}: extensions.policybench_v0_1 must be an object"
        )
    for field in ("policy", "blueprint", "generation", "validation", "grouping"):
        if not isinstance(value.get(field), Mapping):
            raise PolicyBenchReviewError(
                f"record {record.get('id')!r}: policybench extension {field} must be an object"
            )
    return value


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise PolicyBenchReviewError("every review record must have a non-empty string id")
    return value


def _selection_unit(record: Mapping[str, Any]) -> str:
    extension = _extension(record)
    counterfactual = extension.get("counterfactual")
    if isinstance(counterfactual, Mapping):
        group_id = counterfactual.get("counterfactual_group_id")
        if isinstance(group_id, str) and group_id:
            return f"counterfactual:{group_id}"
    return f"record:{_record_id(record)}"


def _coverage(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    extension = _extension(record)
    blueprint = extension["blueprint"]
    assert isinstance(blueprint, Mapping)
    result: set[tuple[str, str]] = set()
    for field in ("domain", "language", "category"):
        value = blueprint.get(field)
        if isinstance(value, str) and value:
            result.add((field, value))
    annotations = record.get("annotations")
    if isinstance(annotations, Mapping):
        for field in (
            "instruction_presentation",
            "user_goal_alignment",
            "protected_policy_alignment",
            "authority_status",
        ):
            value = annotations.get(field)
            if isinstance(value, str) and value:
                result.add((field, value))
    content = record.get("content")
    if isinstance(content, Mapping) and isinstance(content.get("source_role"), str):
        result.add(("source_role", content["source_role"]))
    counterfactual = extension.get("counterfactual")
    counterfactual_type = (
        counterfactual.get("counterfactual_type") if isinstance(counterfactual, Mapping) else "NONE"
    )
    if isinstance(counterfactual_type, str) and counterfactual_type:
        result.add(("counterfactual_type", counterfactual_type))
    return result


def _stratum_value(record: Mapping[str, Any], field: str) -> str:
    extension = _extension(record)
    if field in {"domain", "language", "category"}:
        blueprint = extension["blueprint"]
        assert isinstance(blueprint, Mapping)
        value = blueprint.get(field)
    elif field == "source_role":
        content = record.get("content")
        value = content.get(field) if isinstance(content, Mapping) else None
    elif field == "counterfactual_type":
        counterfactual = extension.get("counterfactual")
        value = (
            counterfactual.get("counterfactual_type")
            if isinstance(counterfactual, Mapping)
            else "NONE"
        )
    else:
        annotations = record.get("annotations")
        value = annotations.get(field) if isinstance(annotations, Mapping) else None
    return value if isinstance(value, str) and value else "UNKNOWN"


def _distribution(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[_stratum_value(record, field)] += 1
    return {name: counts[name] for name in sorted(counts)}


def _eligible(record: Mapping[str, Any]) -> bool:
    extension = _extension(record)
    validation = extension["validation"]
    assert isinstance(validation, Mapping)
    return (
        extension.get("data_quality") in {"SILVER_TEMPLATE", "SILVER_GENERATED", "SILVER_VALIDATED"}
        and extension.get("human_validation_status") == "PENDING"
        and extension.get("dataset_split") != "EXCLUDED"
        and validation.get("overall_status") == "PASSED"
    )


def select_review_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    record_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministic candidates while keeping counterfactual siblings atomic."""

    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 1:
        raise PolicyBenchReviewError("record_count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PolicyBenchReviewError("seed must be an integer")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise PolicyBenchReviewError("every review record must be an object")
        record_id = _record_id(record)
        if record_id in by_id:
            raise PolicyBenchReviewError(f"duplicate canonical record id: {record_id}")
        extension = _extension(record)
        if (
            extension.get("data_quality") == "GOLD_HUMAN_CONFIRMED"
            or extension.get("human_validation_status") != "PENDING"
        ):
            raise PolicyBenchReviewError(
                f"record {record_id!r}: PolicyBench review input must never claim GOLD"
            )
        by_id[record_id] = record

    eligible = [record for record in by_id.values() if _eligible(record)]
    designated = [
        record
        for record in eligible
        if _extension(record).get("dataset_split") == "human_review_candidates"
    ]
    use_designated_pool = len(designated) >= record_count
    pool = designated if use_designated_pool else eligible
    if not pool:
        raise PolicyBenchReviewError("no validated SILVER records are eligible for review")

    units: dict[str, list[Mapping[str, Any]]] = {}
    for record in sorted(pool, key=_record_id):
        units.setdefault(_selection_unit(record), []).append(record)
    for members in units.values():
        members.sort(key=_record_id)
    coverage_by_unit = {
        unit_id: set().union(*(_coverage(record) for record in members))
        for unit_id, members in units.items()
    }
    uncovered = set().union(*coverage_by_unit.values())
    selected_units: list[str] = []
    selected_count = 0

    while uncovered:
        candidates = []
        for unit_id, members in units.items():
            if unit_id in selected_units:
                continue
            size = len(members)
            new_coverage = len(coverage_by_unit[unit_id].intersection(uncovered))
            if new_coverage == 0:
                continue
            if selected_count and selected_count + size > record_count:
                continue
            candidates.append(
                (
                    -new_coverage,
                    size,
                    sha256_text(f"{seed}:coverage:{unit_id}"),
                    unit_id,
                )
            )
        if not candidates:
            break
        unit_id = min(candidates)[-1]
        selected_units.append(unit_id)
        selected_count += len(units[unit_id])
        uncovered.difference_update(coverage_by_unit[unit_id])
        if selected_count >= record_count:
            break

    remaining_units = sorted(
        (unit_id for unit_id in units if unit_id not in selected_units),
        key=lambda unit_id: (sha256_text(f"{seed}:fill:{unit_id}"), unit_id),
    )
    for unit_id in remaining_units:
        size = len(units[unit_id])
        if selected_count + size > record_count:
            continue
        selected_units.append(unit_id)
        selected_count += size
        if selected_count >= record_count:
            break

    if not selected_units:
        first = min(
            units,
            key=lambda unit_id: (
                len(units[unit_id]),
                sha256_text(f"{seed}:fallback:{unit_id}"),
                unit_id,
            ),
        )
        selected_units.append(first)
    selected = [dict(record) for unit_id in selected_units for record in units[unit_id]]
    selected.sort(key=_record_id)
    selected_ids = [_record_id(record) for record in selected]
    deficiencies = []
    if len(selected) < record_count:
        deficiencies.append("group_atomic_selection_below_requested_count")
    if len(selected) > record_count:
        deficiencies.append("smallest_atomic_group_exceeds_requested_count")
    if uncovered:
        deficiencies.append("not_all_domain_language_category_values_covered")
    report = {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "gold_claim_permitted": False,
        "seed": seed,
        "requested_records": record_count,
        "eligible_records": len(pool),
        "eligible_pool": (
            "human_review_candidates" if use_designated_pool else "all_validated_silver_records"
        ),
        "selected_records": len(selected),
        "selected_atomic_groups": len(selected_units),
        "selected_ids": selected_ids,
        "distributions": {
            field: _distribution(selected, field)
            for field in (
                "domain",
                "language",
                "category",
                "instruction_presentation",
                "user_goal_alignment",
                "protected_policy_alignment",
                "authority_status",
                "source_role",
                "counterfactual_type",
            )
        },
        "deficiencies": deficiencies,
        "selection_rule": (
            "deterministic greedy context/label/counterfactual stratum coverage, then "
            "SHA-256 fill; "
            "counterfactual groups remain atomic and ordinary records remain independently "
            "selectable"
        ),
    }
    return selected, report


def _blinded_record(record: Mapping[str, Any], blind_id: str) -> dict[str, Any]:
    content = record.get("content")
    context = record.get("context")
    if not isinstance(content, Mapping) or not isinstance(context, Mapping):
        raise PolicyBenchReviewError(
            f"record {_record_id(record)!r}: content and context must be objects"
        )
    return {
        "blind_id": blind_id,
        "content": {field: content.get(field) for field in _CONTENT_FIELDS},
        "context": {
            field: context.get(field)
            for field in ("protected_policy", "user_goal", "available_capabilities")
        },
        "annotation": {
            field: [] if field in {"attack_families", "attack_objectives", "spans"} else None
            for field in _ANNOTATION_FIELDS
        },
        "notes": None,
    }


def create_review_packets(
    records: Iterable[Mapping[str, Any]],
    output: str | Path,
    *,
    record_count: int,
    seed: int,
) -> ReviewPacketResult:
    """Create A/B packets, hidden manifests, selection report, and checksums."""

    selected, selection_report = select_review_candidates(
        records,
        record_count=record_count,
        seed=seed,
    )
    root = Path(output)
    blinded: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(selected, key=_record_id), start=1):
        blind_id = f"candidate_{index:04d}"
        blinded.append(_blinded_record(record, blind_id))
        extension = _extension(record)
        policy = extension["policy"]
        blueprint = extension["blueprint"]
        grouping = extension["grouping"]
        assert isinstance(policy, Mapping)
        assert isinstance(blueprint, Mapping)
        assert isinstance(grouping, Mapping)
        counterfactual = extension.get("counterfactual")
        hidden.append(
            {
                "blind_id": blind_id,
                "canonical_id": _record_id(record),
                "policy_id": policy.get("policy_id"),
                "scenario_blueprint_id": blueprint.get("scenario_blueprint_id"),
                "counterfactual_group_id": (
                    counterfactual.get("counterfactual_group_id")
                    if isinstance(counterfactual, Mapping)
                    else None
                ),
                "semantic_duplicate_cluster_id": grouping.get("semantic_duplicate_cluster_id"),
                "split_group_id": grouping.get("split_group_id"),
                "dataset_split": extension.get("dataset_split"),
                "data_quality": extension.get("data_quality"),
                "human_validation_status": extension.get("human_validation_status"),
            }
        )

    packet_info: dict[str, dict[str, Any]] = {}
    for annotator in ("A", "B"):
        ordered = sorted(
            blinded,
            key=lambda item: (
                sha256_text(f"{seed}:{annotator}:{item['blind_id']}"),
                item["blind_id"],
            ),
        )
        name = f"annotator_{annotator}.jsonl"
        path = root / name
        write_jsonl(path, ordered)
        packet_info[annotator] = {
            "path": name,
            "records": len(ordered),
            "sha256": sha256_file(path),
        }

    write_json(root / "selection_report.json", selection_report)
    researcher_manifest = {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "gold_claim_permitted": False,
        "packets": packet_info,
        "records": hidden,
    }
    write_json(root / "researcher_manifest.json", researcher_manifest)
    packet_manifest = {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "gold_claim_permitted": False,
        "records": len(selected),
        "packets": packet_info,
        "selection_report_sha256": sha256_file(root / "selection_report.json"),
        "researcher_manifest_sha256": sha256_file(root / "researcher_manifest.json"),
        "human_validation_status": "PENDING",
        "automatic_gold_records": 0,
    }
    write_json(root / "packet_manifest.json", packet_manifest)
    checksum_names = [
        "annotator_A.jsonl",
        "annotator_B.jsonl",
        "packet_manifest.json",
        "researcher_manifest.json",
        "selection_report.json",
    ]
    write_named_checksums(root, checksum_names)
    return ReviewPacketResult(
        output=root,
        selected_ids=tuple(selection_report["selected_ids"]),
        selection_report=selection_report,
        packet_manifest=packet_manifest,
    )


create_policybench_review_packets = create_review_packets


__all__ = [
    "PolicyBenchReviewError",
    "ReviewPacketResult",
    "create_policybench_review_packets",
    "create_review_packets",
    "select_review_candidates",
]
