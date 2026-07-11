"""Conservative mapping-quality assessment and review-queue construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from numbers import Real
from typing import Any

ANNOTATION_TIERS = frozenset(
    {
        "GOLD_SOURCE",
        "DETERMINISTIC_MAPPING",
        "HEURISTIC_MAPPING",
        "UNANNOTATED",
    }
)

_PROFILE_REVIEW_STATUSES = frozenset({"NEEDS_REVIEW", "UNMAPPED", "EXCLUDED"})
_ANNOTATION_REVIEW_STATUSES = frozenset({"INSUFFICIENT_CONTEXT", "DISAGREEMENT", "EXCLUDED"})
_HASH_FIELDS = ("raw_hash", "normalized_hash", "contextual_hash")
_CONTENT_PREVIEW_LENGTH = 240


def _nested_mapping(value: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _source_id(record: Mapping[str, Any]) -> str:
    source = _nested_mapping(
        record,
        "metadata",
        "dataset_provenance",
        "source_dataset",
    ).get("id")
    return source if isinstance(source, str) and source else "UNKNOWN"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _profile_for_source(
    profiles: Mapping[str, Mapping[str, Any]], source_id: str
) -> Mapping[str, Any] | None:
    profile = profiles.get(source_id)
    if profile is None:
        return None
    if not isinstance(profile, Mapping):
        raise TypeError(f"mapping profile for source {source_id!r} must be an object")
    return profile


def _mapping_confidence(profile: Mapping[str, Any], source_id: str) -> float:
    value = profile.get("mapping_confidence")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"mapping profile for source {source_id!r} must define a numeric mapping_confidence"
        )
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"mapping_confidence for source {source_id!r} must be between 0 and 1")
    return confidence


def assess_mapping(
    record: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assess a record using an explicit, source-indexed mapping profile.

    A missing source profile is deliberately treated as unannotated instead of
    inferring human-equivalent quality from a source label or annotator confidence.
    Existing provenance review flags remain authoritative and can only make the
    result more conservative.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be an object")
    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be an object indexed by source id")

    source_id = _source_id(record)
    profile = _profile_for_source(profiles, source_id)
    provenance_mapping = _nested_mapping(
        record,
        "metadata",
        "dataset_provenance",
        "mapping",
    )
    annotations = _nested_mapping(record, "annotations")

    if profile is None:
        tier = "UNANNOTATED"
        confidence = 0.0
        requires_manual_review = True
        reasons: list[Any] = ["NO_SOURCE_MAPPING_PROFILE"]
    else:
        tier = profile.get("annotation_tier")
        if tier not in ANNOTATION_TIERS:
            allowed = ", ".join(sorted(ANNOTATION_TIERS))
            raise ValueError(f"annotation_tier for source {source_id!r} must be one of: {allowed}")
        confidence = _mapping_confidence(profile, source_id)
        configured_review = profile.get("requires_manual_review", False)
        if not isinstance(configured_review, bool):
            raise ValueError(f"requires_manual_review for source {source_id!r} must be a boolean")
        requires_manual_review = configured_review
        configured_reasons = profile.get("review_reasons", [])
        if not isinstance(configured_reasons, list):
            raise ValueError(f"review_reasons for source {source_id!r} must be an array")
        reasons = list(configured_reasons)

    mapping_status = provenance_mapping.get("status")
    if mapping_status in _PROFILE_REVIEW_STATUSES:
        requires_manual_review = True
        reasons.append(f"PROVENANCE_MAPPING_STATUS_{mapping_status}")
    elif mapping_status not in {None, "DETERMINISTIC"}:
        requires_manual_review = True
        reasons.append("PROVENANCE_MAPPING_STATUS_UNKNOWN")
    elif mapping_status is None:
        requires_manual_review = True
        reasons.append("PROVENANCE_MAPPING_STATUS_MISSING")

    unmapped_labels = provenance_mapping.get("unmapped_labels", [])
    if isinstance(unmapped_labels, list) and unmapped_labels:
        requires_manual_review = True
        reasons.append("UNMAPPED_SOURCE_LABELS")

    provenance_reasons = provenance_mapping.get("review_reasons", [])
    if isinstance(provenance_reasons, list) and provenance_reasons:
        requires_manual_review = True
        reasons.extend(provenance_reasons)

    annotation_status = annotations.get("annotation_status")
    if annotation_status in _ANNOTATION_REVIEW_STATUSES:
        requires_manual_review = True
        reasons.append(f"ANNOTATION_STATUS_{annotation_status}")

    if tier == "UNANNOTATED":
        confidence = 0.0
        requires_manual_review = True

    return {
        "annotation_tier": tier,
        "mapping_confidence": confidence,
        "requires_manual_review": requires_manual_review,
        "review_reasons": _unique_strings(reasons),
    }


def _quality_extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested_mapping(record, "extensions", "quality_v0_1")


def _hashes(record: Mapping[str, Any]) -> dict[str, Any]:
    quality = _quality_extension(record)
    candidates = (
        quality.get("hashes"),
        quality.get("deduplication"),
        quality,
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        selected = {field: candidate[field] for field in _HASH_FIELDS if field in candidate}
        if selected:
            return selected
    return {}


def build_review_queue(
    records: Iterable[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Return deterministic review entries for uncertain mappings or axes."""

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a number between 0 and 1")
    threshold_value = float(threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    queue: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("every record must be an object")
        quality = _quality_extension(record)
        mapping_quality = quality.get("mapping_quality")
        mapping_quality = mapping_quality if isinstance(mapping_quality, Mapping) else {}
        annotations = _nested_mapping(record, "annotations")
        conditions: list[str] = []

        confidence = mapping_quality.get("mapping_confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            conditions.append("MAPPING_CONFIDENCE_MISSING")
        elif float(confidence) < threshold_value:
            conditions.append("MAPPING_CONFIDENCE_BELOW_THRESHOLD")

        if mapping_quality.get("requires_manual_review") is True:
            conditions.append("REQUIRES_MANUAL_REVIEW")
        if annotations.get("authority_status") == "UNKNOWN":
            conditions.append("AUTHORITY_STATUS_UNKNOWN")
        if annotations.get("user_goal_alignment") == "UNDETERMINED":
            conditions.append("USER_GOAL_ALIGNMENT_UNDETERMINED")
        if annotations.get("protected_policy_alignment") == "UNDETERMINED":
            conditions.append("PROTECTED_POLICY_ALIGNMENT_UNDETERMINED")

        if not conditions:
            continue

        content = _nested_mapping(record, "content")
        context = _nested_mapping(record, "context")
        provenance = _nested_mapping(record, "metadata", "dataset_provenance")
        grouping = quality.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else {}
        deduplication = quality.get("deduplication")
        deduplication = deduplication if isinstance(deduplication, Mapping) else {}
        text = content.get("text")
        preview = text[:_CONTENT_PREVIEW_LENGTH] if isinstance(text, str) else ""
        record_id = record.get("id")
        queue.append(
            {
                "id": record_id if isinstance(record_id, str) else "UNKNOWN",
                "source": _source_id(record),
                "conditions": conditions,
                "annotations": deepcopy(dict(annotations)),
                "context": deepcopy(dict(context)),
                "content": deepcopy(dict(content)),
                "dataset_provenance": deepcopy(dict(provenance)),
                "mapping_quality": deepcopy(dict(mapping_quality)),
                "grouping": deepcopy(dict(grouping)),
                "deduplication": deepcopy(dict(deduplication)),
                "hashes": deepcopy(_hashes(record)),
                "content_preview": preview,
            }
        )

    return sorted(queue, key=lambda entry: (entry["source"], entry["id"]))


__all__ = ["ANNOTATION_TIERS", "assess_mapping", "build_review_queue"]
