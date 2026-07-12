"""Deterministic quality review and blinded human-review preparation.

This module never creates a gold annotation.  It produces auditable candidate
packets and empty annotation forms; a later adjudication step must supply human
annotations before any record can enter a gold release.
"""

# The review packet structures intentionally keep several schema-shaped literals on one
# line; suppress only mechanical line-length/style diagnostics for this adapter module.
# ruff: noqa: E501, E702

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from promptsec.data.hashing import sha256_file
from promptsec.data.validation import validate_record

PHASE_STATES = (
    "READY_FOR_HUMAN_REVIEW",
    "ANNOTATION_IN_PROGRESS",
    "READY_FOR_ADJUDICATION",
    "GOLD_LOCAL_ONLY",
    "GOLD_REDISTRIBUTABLE",
    "BLOCKED_BY_LICENSE",
    "BLOCKED_BY_ANNOTATION_QUALITY",
)
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


class GoldReviewError(ValueError):
    """Raised when review preparation inputs are incomplete or malformed."""


def load_gold_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GoldReviewError("gold subset configuration must be an object")
    required = {"schema_version", "release_id", "seed", "target_records", "output_dir"}
    missing = sorted(required - set(value))
    if missing:
        raise GoldReviewError(f"gold subset configuration missing keys: {missing}")
    if str(value["schema_version"]) != "0.1":
        raise GoldReviewError("unsupported gold subset schema version")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise GoldReviewError("gold subset seed must be an integer")
    if not isinstance(value["target_records"], int) or value["target_records"] < 1:
        raise GoldReviewError("gold subset target_records must be positive")
    return value


def iter_release_records(release: str | Path) -> list[dict[str, Any]]:
    root = Path(release)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        if path.name in {"review_queue.jsonl", "agentic_review_queue.jsonl"}:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GoldReviewError(f"{path}:{line_number}: expected an object")
            records.append(value)
    if not records:
        raise GoldReviewError(f"release contains no split records: {root}")
    return records


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _source(record: Mapping[str, Any]) -> str:
    value = _nested(record, "metadata", "dataset_provenance", "source_dataset", "id")
    return value if isinstance(value, str) and value else "UNKNOWN"


def _quality(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(record, "extensions", "quality_v0_1")
    return value if isinstance(value, Mapping) else {}


def _hashes(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _quality(record).get("hashes")
    return value if isinstance(value, Mapping) else {}


def _cluster(record: Mapping[str, Any]) -> str:
    value = _hashes(record).get("semantic_cluster_id")
    return value if isinstance(value, str) and value else f"record:{record.get('id', 'UNKNOWN')}"


def audit_corpus(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    ids = [record.get("id") for record in materialized]
    issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    id_counts = Counter(value for value in ids if isinstance(value, str))
    duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
    if duplicate_ids:
        issues["duplicate_canonical_id"] = [{"id": value} for value in duplicate_ids]

    for record in materialized:
        record_id = record.get("id", "UNKNOWN")
        provenance = _nested(record, "metadata", "dataset_provenance")
        if not isinstance(provenance, Mapping):
            issues["missing_provenance"].append({"id": record_id})
        source_record = provenance.get("source_record") if isinstance(provenance, Mapping) else None
        if not isinstance(source_record, Mapping) or not source_record.get("id"):
            issues["invalid_source_record_id"].append({"id": record_id})

        errors = validate_record(record)
        if errors:
            issues["schema_or_structural_error"].append({"id": record_id, "errors": errors})

        annotations = record.get("annotations") if isinstance(record, Mapping) else {}
        annotations = annotations if isinstance(annotations, Mapping) else {}
        context = record.get("context") if isinstance(record, Mapping) else {}
        context = context if isinstance(context, Mapping) else {}
        content = record.get("content") if isinstance(record, Mapping) else {}
        content = content if isinstance(content, Mapping) else {}
        spans = annotations.get("spans", [])
        text = content.get("text", "")
        if isinstance(spans, list) and isinstance(text, str):
            for span in spans:
                if not isinstance(span, Mapping) or not (
                    isinstance(span.get("start"), int)
                    and isinstance(span.get("end"), int)
                    and 0 <= span["start"] < span["end"] <= len(text)
                ):
                    issues["span_error"].append({"id": record_id, "span": span})
        if annotations.get("instruction_presence") == "NO_INSTRUCTION" and (
            annotations.get("attack_families") or annotations.get("attack_objectives")
        ):
            issues["no_instruction_with_attack_labels"].append({"id": record_id})
        if annotations.get("instruction_presence") == "NO_INSTRUCTION":
            expected_na = (
                "instruction_presentation",
                "instruction_addressee",
                "user_goal_alignment",
                "protected_policy_alignment",
                "authority_status",
            )
            if any(annotations.get(field) != "NOT_APPLICABLE" for field in expected_na):
                issues["not_applicable_inconsistent_with_no_instruction"].append({"id": record_id})
        if annotations.get("instruction_presence") == "INSTRUCTION_PRESENT" and any(
            annotations.get(field) == "NOT_APPLICABLE"
            for field in ("instruction_presentation", "instruction_addressee")
        ):
            issues["not_applicable_inconsistent_with_instruction"].append({"id": record_id})
        if annotations.get("attack_objectives") and not annotations.get("attack_families"):
            issues["objective_without_family"].append({"id": record_id})
        if annotations.get("user_goal_alignment") in {"ALIGNED", "MISALIGNED"} and not (
            isinstance(context.get("user_goal"), str) and context.get("user_goal")
        ):
            issues["alignment_without_user_goal"].append({"id": record_id})
        if annotations.get("protected_policy_alignment") in {"COMPLIANT", "CONFLICTING"} and not (
            isinstance(context.get("protected_policy"), str) and context.get("protected_policy")
        ):
            issues["policy_alignment_without_policy"].append({"id": record_id})
        if annotations.get("authority_status") == "SPOOFED" and not any(
            isinstance(span, Mapping) and span.get("type") == "AUTHORITY_CLAIM" for span in spans
        ):
            issues["spoofed_without_authority_claim"].append({"id": record_id})
        if annotations.get("authority_status") == "WITHIN_AUTHORITY" and not (
            isinstance(context.get("user_goal"), str) and context.get("user_goal")
        ):
            issues["within_authority_without_boundary"].append({"id": record_id})
        mapping = _quality(record).get("mapping_quality")
        mapping = mapping if isinstance(mapping, Mapping) else {}
        evidence = _nested(record, "extensions", "mapping_evidence")
        provenance_mapping = _nested(record, "metadata", "dataset_provenance", "mapping")
        has_rationale = isinstance(evidence, Mapping) and evidence.get("mapping_version")
        has_rationale = bool(
            has_rationale
            or (
                isinstance(provenance_mapping, Mapping)
                and provenance_mapping.get("ruleset")
                and provenance_mapping.get("field_mappings")
            )
        )
        if not mapping or not isinstance(mapping.get("mapping_confidence"), (int, float)):
            issues["missing_mapping_confidence"].append({"id": record_id})
        if not has_rationale:
            issues["missing_mapping_rationale"].append({"id": record_id})
        if mapping.get("requires_manual_review") is not True and _source(record) in {
            "bipia",
            "open_prompt_injection",
            "notinject",
            "injecagent",
            "agentdojo",
        }:
            issues["source_mapping_not_flagged_for_review"].append({"id": record_id})
        if annotations.get("annotation_status") == "CONFIRMED" and _source(record) not in {
            "promptinject"
        }:
            issues["source_derived_marked_confirmed"].append({"id": record_id})

    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_normalized_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in materialized:
        by_cluster[_cluster(record)].append(record)
        normalized_hash = _hashes(record).get("normalized_hash")
        if isinstance(normalized_hash, str) and normalized_hash:
            by_normalized_hash[normalized_hash].append(record)
    cluster_conflicts = []
    for cluster_id, members in sorted(by_cluster.items()):
        verdicts = {str(_nested(item, "derived", "prompt_injection_verdict")) for item in members}
        families = {
            tuple(_nested(item, "annotations", "attack_families") or []) for item in members
        }
        if len(verdicts) > 1 or len(families) > 1:
            cluster_conflicts.append(
                {
                    "semantic_cluster_id": cluster_id,
                    "record_ids": sorted(str(item.get("id")) for item in members),
                    "verdicts": sorted(verdicts),
                    "families": sorted(families),
                }
            )
    if cluster_conflicts:
        issues["semantic_cluster_conflict"] = cluster_conflicts
    contextual_conflicts = []
    for normalized_hash, members in sorted(by_normalized_hash.items()):
        contexts = {_hashes(item).get("contextual_hash") for item in members}
        if len(contexts) > 1:
            contextual_conflicts.append(
                {
                    "normalized_hash": normalized_hash,
                    "record_ids": sorted(str(item.get("id")) for item in members),
                    "contextual_hashes": sorted(str(item) for item in contexts),
                }
            )
    if contextual_conflicts:
        issues["contextual_conflict"] = contextual_conflicts

    return {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "records": len(materialized),
        "sources": dict(sorted(Counter(_source(item) for item in materialized).items())),
        "semantic_clusters": len(by_cluster),
        "issue_counts": {key: len(value) for key, value in sorted(issues.items())},
        "issues": {key: value for key, value in sorted(issues.items())},
        "human_annotation_files_present": False,
        "gold_claim_permitted": False,
    }


def _score(
    record: Mapping[str, Any], frequencies: Mapping[str, Counter[str]]
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    structural_errors = validate_record(record)
    if structural_errors:
        score += 100
        reasons.append("structural_validation_error")
    quality = _quality(record)
    mapping = (
        quality.get("mapping_quality")
        if isinstance(quality.get("mapping_quality"), Mapping)
        else {}
    )
    annotations = (
        record.get("annotations") if isinstance(record.get("annotations"), Mapping) else {}
    )
    content = record.get("content") if isinstance(record.get("content"), Mapping) else {}
    agentic = _nested(record, "extensions", "agentic_source")
    if (
        mapping.get("requires_manual_review") is True
        or float(mapping.get("mapping_confidence", 0)) < 0.85
    ):
        score += 20
        reasons.append("heuristic_or_low_confidence_mapping")
    for field in ("authority_status", "user_goal_alignment", "protected_policy_alignment"):
        if annotations.get(field) in {"UNKNOWN", "UNDETERMINED"}:
            score += 8
            reasons.append(f"uncertain_{field}")
    if not content.get("text"):
        score += 30
        reasons.append("missing_content")
    if isinstance(agentic, Mapping) and agentic:
        score += 6
        reasons.append("agentic_scenario")
    if _source(record) == "notinject":
        score += 7
        reasons.append("hard_negative_coverage")
    if annotations.get("instruction_presentation") in {"QUOTED_OR_REPORTED", "HYPOTHETICAL"}:
        score += 6
        reasons.append("quoted_or_hypothetical")
    if annotations.get("authority_status") == "SPOOFED":
        score += 5
        reasons.append("claimed_authority")
    spans = annotations.get("spans") if isinstance(annotations.get("spans"), list) else []
    if len(spans) > 1:
        score += 3
        reasons.append("multiple_spans")
    if len(str(content.get("text", ""))) > 500:
        score += 3
        reasons.append("long_context")
    source = _source(record)
    if frequencies["source"][source] <= 40:
        score += 4
        reasons.append("rare_source")
    family = tuple(annotations.get("attack_families") or ["NONE"])
    if frequencies["family"][family] <= 20:
        score += 4
        reasons.append("rare_attack_family")
    objective = tuple(annotations.get("attack_objectives") or ["NONE"])
    if frequencies["objective"][objective] <= 20:
        score += 4
        reasons.append("rare_attack_objective")
    if source in {"bipia", "injecagent", "agentdojo"}:
        score += 3
        reasons.append("license_sensitive_source")
    band = (
        "P0_BLOCKING_ERROR"
        if structural_errors
        or score >= 35
        and (not content.get("text") or not _nested(record, "metadata", "dataset_provenance"))
        else (
            "P1_GOLD_CANDIDATE"
            if score >= 25
            else "P2_MAPPING_AUDIT"
            if score >= 15
            else "P3_COVERAGE"
            if score >= 8
            else "P4_LOW_PRIORITY"
        )
    )
    reasons.insert(0, band)
    return score, sorted(set(reasons))


def review_priorities(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(records)
    frequencies = {
        "source": Counter(_source(item) for item in materialized),
        "family": Counter(
            tuple(_nested(item, "annotations", "attack_families") or ["NONE"])
            for item in materialized
        ),
        "objective": Counter(
            tuple(_nested(item, "annotations", "attack_objectives") or ["NONE"])
            for item in materialized
        ),
    }
    result = []
    for record in materialized:
        score, reasons = _score(record, frequencies)
        result.append(
            {
                "id": record.get("id", "UNKNOWN"),
                "source": _source(record),
                "score": score,
                "priority_band": reasons[0],
                "reasons": reasons[1:],
                "semantic_cluster_id": _cluster(record),
            }
        )
    return sorted(result, key=lambda item: (-item["score"], item["id"]))


def select_candidates(
    records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    materialized = list(records)
    target = int(config["target_records"])
    seed = int(config["seed"])
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in materialized:
        by_cluster[_cluster(record)].append(record)

    def rank(record: Mapping[str, Any], salt: str = "") -> str:
        return hashlib.sha256(f"{seed}:{salt}:{record.get('id')}".encode()).hexdigest()

    selected: dict[str, Mapping[str, Any]] = {}
    selected_reasons: dict[str, list[str]] = {}
    protected_ids: set[str] = set()
    small_limit = int(config.get("include_all_sources_with_at_most", 40))
    source_counts = Counter(_source(item) for item in materialized)
    for source, count in sorted(source_counts.items()):
        if count <= small_limit:
            for record in materialized:
                if _source(record) == source:
                    selected[str(record["id"])] = record
                    selected_reasons[str(record["id"])] = ["small_source_complete"]
                    protected_ids.add(str(record["id"]))
    cluster_choices: dict[str, Mapping[str, Any]] = {
        cluster_id: min(members, key=lambda item: rank(item, cluster_id))
        for cluster_id, members in by_cluster.items()
    }
    # Preserve explicit coverage quotas while keeping at most one ordinary
    # representative per semantic cluster.
    for source, minimum in (
        ("notinject", int(config.get("minimum_hard_negative_records", 0))),
        ("agentic", int(config.get("minimum_agentic_records", 0))),
    ):
        source_members = [
            item
            for item in materialized
            if (
                _source(item) == source
                if source == "notinject"
                else _source(item) in {"injecagent", "agentdojo"}
            )
        ]
        source_clusters = sorted({_cluster(item) for item in source_members})
        for cluster_id in sorted(
            source_clusters, key=lambda value: rank(cluster_choices[value], f"quota:{source}")
        )[:minimum]:
            chosen_members = [
                item
                for item in by_cluster[cluster_id]
                if _source(item)
                in ({"injecagent", "agentdojo"} if source == "agentic" else {source})
            ]
            chosen = min(chosen_members, key=lambda item: rank(item, f"quota:{source}"))
            selected.setdefault(str(chosen["id"]), chosen)
            selected_reasons.setdefault(str(chosen["id"]), []).append(f"quota_{source}")
            protected_ids.add(str(chosen["id"]))
    for _cluster_id, chosen in sorted(cluster_choices.items()):
        selected.setdefault(str(chosen["id"]), chosen)
        selected_reasons.setdefault(str(chosen["id"]), []).append("semantic_cluster_representative")
    if len(selected) > target:
        # Keep quota and small-source records, then fill deterministically.
        if len(protected_ids) > target:
            raise GoldReviewError("coverage quotas exceed gold subset target")
        ranked_ids = sorted(
            (value for value in selected if value not in protected_ids),
            key=lambda value: rank(selected[value], "final"),
        )
        keep_ids = protected_ids | set(ranked_ids[: target - len(protected_ids)])
        selected = {key: value for key, value in selected.items() if key in keep_ids}
    ranked = sorted(materialized, key=lambda item: rank(item, "fill"))
    for record in ranked:
        if len(selected) >= target:
            break
        record_id = str(record["id"])
        if record_id not in selected and _cluster(record) not in {
            _cluster(item) for item in selected.values()
        }:
            selected[record_id] = record
            selected_reasons[record_id] = ["coverage_fill"]
    selected_records = sorted(selected.values(), key=lambda item: rank(item, "output"))[:target]
    selected_ids = {str(item["id"]) for item in selected_records}
    source_distribution = Counter(_source(item) for item in selected_records)
    agentic_count = sum(
        1 for item in selected_records if _source(item) in {"injecagent", "agentdojo"}
    )
    hard_negative_count = source_distribution["notinject"]
    report = {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "seed": seed,
        "release_id": config.get("release_id"),
        "target_records": target,
        "selected_records": len(selected_records),
        "selected_ids": sorted(selected_ids),
        "source_distribution": dict(sorted(source_distribution.items())),
        "semantic_clusters_selected": len({_cluster(item) for item in selected_records}),
        "agentic_records": agentic_count,
        "hard_negative_records": hard_negative_count,
        "deficiencies": [
            reason
            for ok, reason in (
                (
                    agentic_count >= int(config.get("minimum_agentic_records", 0)),
                    "minimum_agentic_records",
                ),
                (
                    hard_negative_count >= int(config.get("minimum_hard_negative_records", 0)),
                    "minimum_hard_negative_records",
                ),
            )
            if not ok
        ],
        "selection_rule": "deterministic SHA-256 ranking by fixed seed and semantic_cluster_id; one representative per cluster, then small-source completion and coverage fill",
        "rejected_candidate_count": len(materialized) - len(selected_records),
    }
    return [dict(item) for item in selected_records], report


def create_blinded_packets(
    records: Iterable[Mapping[str, Any]], output: str | Path, *, seed: int
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    selected = list(records)
    hidden: list[dict[str, Any]] = []
    blinded: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(selected, key=lambda item: str(item["id"]))):
        blind_id = f"candidate_{index + 1:04d}"
        hidden.append(
            {
                "blind_id": blind_id,
                "canonical_id": record["id"],
                "source": _source(record),
                "source_record_id": _nested(
                    record, "metadata", "dataset_provenance", "source_record", "id"
                ),
                "semantic_cluster_id": _cluster(record),
                "license_status": "REVIEW_REQUIRED",
            }
        )
        content = record.get("content", {})
        context = record.get("context", {})
        blinded.append(
            {
                "blind_id": blind_id,
                "content": {
                    key: content.get(key)
                    for key in (
                        "text",
                        "language",
                        "delivery_mode",
                        "source_role",
                        "content_origin",
                        "ingestion_path",
                        "modality",
                    )
                },
                "context": {
                    key: context.get(key)
                    for key in ("protected_policy", "user_goal", "available_capabilities")
                },
                "annotation": {
                    field: []
                    if field in {"attack_families", "attack_objectives", "spans"}
                    else None
                    for field in _ANNOTATION_FIELDS
                },
                "notes": None,
            }
        )

    def order(item: Mapping[str, Any], annotator: str) -> str:
        return hashlib.sha256(f"{seed}:{annotator}:{item['blind_id']}".encode()).hexdigest()

    packets = {}
    for annotator in ("A", "B"):
        path = root / f"annotator_{annotator}.jsonl"
        ordered = sorted(blinded, key=lambda item: order(item, annotator))
        path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for item in ordered
            ),
            encoding="utf-8",
            newline="\n",
        )
        packets[annotator] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "records": len(ordered),
        }
    hidden_path = root / "researcher_manifest.json"
    hidden_doc = {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "packets": packets,
        "records": hidden,
    }
    hidden_path.write_text(
        json.dumps(hidden_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "records": len(selected),
        "packets": packets,
        "researcher_manifest_sha256": sha256_file(hidden_path),
    }


__all__ = [
    "PHASE_STATES",
    "GoldReviewError",
    "audit_corpus",
    "create_blinded_packets",
    "iter_release_records",
    "load_gold_config",
    "review_priorities",
    "select_candidates",
]
