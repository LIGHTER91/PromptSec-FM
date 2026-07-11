"""Deterministic descriptive statistics for canonical PromptSec records."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any

_UNKNOWN_FIELDS = (
    "content.delivery_mode",
    "content.source_role",
    "content.content_origin",
    "content.ingestion_path",
    "content.source_integrity",
    "annotations.instruction_presence",
    "annotations.instruction_presentation",
    "annotations.instruction_addressee",
    "annotations.user_goal_alignment",
    "annotations.protected_policy_alignment",
    "annotations.authority_status",
)
_TIERS = (
    "GOLD_SOURCE",
    "DETERMINISTIC_MAPPING",
    "HEURISTIC_MAPPING",
    "UNANNOTATED",
)


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _label(value: Any, default: str = "UNKNOWN") -> str:
    return value if isinstance(value, str) and value else default


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _metric(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "rate": _rate(count, total)}


def _numeric_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    """Summarize values using the deterministic nearest-rank p95."""

    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0,
            "max": 0,
        }

    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        median = float(ordered[middle])
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    return {
        "count": count,
        "min": ordered[0],
        "mean": round(sum(ordered) / count, 6),
        "median": round(float(median), 6),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _quality(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(record, "extensions", "quality_v0_1")
    return value if isinstance(value, Mapping) else {}


def _mapping_quality(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _quality(record).get("mapping_quality")
    return value if isinstance(value, Mapping) else {}


def _increment_multi_label(counter: Counter[str], value: Any, *, empty_label: str) -> None:
    if not isinstance(value, list):
        counter[empty_label] += 1
        return
    labels = sorted({item for item in value if isinstance(item, str) and item})
    if not labels:
        counter[empty_label] += 1
        return
    counter.update(labels)


def _source(record: Mapping[str, Any]) -> str:
    return _label(_nested(record, "metadata", "dataset_provenance", "source_dataset", "id"))


def _is_review_candidate(
    record: Mapping[str, Any],
    mapping_quality: Mapping[str, Any],
    review_threshold: float | None,
) -> bool:
    annotations = _nested(record, "annotations")
    annotations = annotations if isinstance(annotations, Mapping) else {}
    confidence = mapping_quality.get("mapping_confidence")
    confidence_missing = not (
        isinstance(confidence, Real)
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0
    )
    below_threshold = bool(
        review_threshold is not None
        and not confidence_missing
        and float(confidence) < review_threshold
    )
    return bool(
        confidence_missing
        or below_threshold
        or mapping_quality.get("requires_manual_review") is True
        or annotations.get("authority_status") == "UNKNOWN"
        or annotations.get("user_goal_alignment") == "UNDETERMINED"
        or annotations.get("protected_policy_alignment") == "UNDETERMINED"
    )


def compute_statistics(
    records: Iterable[Mapping[str, Any]], *, review_threshold: float | None = None
) -> dict[str, Any]:
    """Compute corpus distributions and quality indicators without ML dependencies."""

    if review_threshold is not None and (
        isinstance(review_threshold, bool)
        or not isinstance(review_threshold, Real)
        or not 0 <= float(review_threshold) <= 1
    ):
        raise ValueError("review_threshold must be a number between 0 and 1")
    threshold = float(review_threshold) if review_threshold is not None else None

    materialized = list(records)
    if not all(isinstance(record, Mapping) for record in materialized):
        raise TypeError("every record must be an object")
    total = len(materialized)

    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    attack_families: Counter[str] = Counter()
    attack_objectives: Counter[str] = Counter()
    delivery_modes: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    span_types: Counter[str] = Counter()
    unknown_counts: dict[str, Counter[str]] = {
        field: Counter({"UNKNOWN": 0, "UNDETERMINED": 0}) for field in _UNKNOWN_FIELDS
    }

    character_lengths: list[int] = []
    utf8_lengths: list[int] = []
    span_counts: list[int] = []
    span_lengths: list[int] = []
    mapping_confidences: list[float] = []
    missing_mapping_confidence = 0
    records_with_spans = 0
    manual_review_count = 0
    review_candidate_count = 0

    for record in materialized:
        content = _nested(record, "content")
        content = content if isinstance(content, Mapping) else {}
        annotations = _nested(record, "annotations")
        annotations = annotations if isinstance(annotations, Mapping) else {}
        quality = _quality(record)
        grouping = quality.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else {}
        mapping_quality = _mapping_quality(record)

        sources[_source(record)] += 1
        languages[_label(content.get("language"))] += 1
        domains[_label(grouping.get("domain"))] += 1
        delivery_modes[_label(content.get("delivery_mode"))] += 1
        _increment_multi_label(
            attack_families,
            annotations.get("attack_families"),
            empty_label="NONE",
        )
        _increment_multi_label(
            attack_objectives,
            annotations.get("attack_objectives"),
            empty_label="NONE",
        )

        tier = _label(mapping_quality.get("annotation_tier"), default="UNANNOTATED")
        if tier not in _TIERS:
            tier = "UNANNOTATED"
        tiers[tier] += 1
        if mapping_quality.get("requires_manual_review") is True:
            manual_review_count += 1
        if _is_review_candidate(record, mapping_quality, threshold):
            review_candidate_count += 1

        confidence = mapping_quality.get("mapping_confidence")
        if (
            isinstance(confidence, Real)
            and not isinstance(confidence, bool)
            and 0.0 <= float(confidence) <= 1.0
        ):
            mapping_confidences.append(float(confidence))
        else:
            missing_mapping_confidence += 1

        text = content.get("text")
        text = text if isinstance(text, str) else ""
        character_lengths.append(len(text))
        utf8_lengths.append(len(text.encode("utf-8")))

        spans = annotations.get("spans", [])
        if not isinstance(spans, list):
            raise TypeError(f"record {_label(record.get('id'))!r} annotations.spans must be a list")
        span_counts.append(len(spans))
        if spans:
            records_with_spans += 1
        for span in spans:
            if not isinstance(span, Mapping):
                raise TypeError(f"record {_label(record.get('id'))!r} contains a non-object span")
            start = span.get("start")
            end = span.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or end < start
            ):
                raise ValueError(
                    f"record {_label(record.get('id'))!r} contains invalid span bounds"
                )
            span_lengths.append(end - start)
            span_types[_label(span.get("type"))] += 1

        for field in _UNKNOWN_FIELDS:
            section, name = field.split(".", 1)
            container = content if section == "content" else annotations
            value = container.get(name)
            if value in {"UNKNOWN", "UNDETERMINED"}:
                unknown_counts[field][value] += 1

    tier_metrics = {tier: _metric(tiers[tier], total) for tier in _TIERS}
    unknown_metrics = {
        field: {
            "UNKNOWN": counts["UNKNOWN"],
            "UNKNOWN_rate": _rate(counts["UNKNOWN"], total),
            "UNDETERMINED": counts["UNDETERMINED"],
            "UNDETERMINED_rate": _rate(counts["UNDETERMINED"], total),
            "combined": _metric(counts["UNKNOWN"] + counts["UNDETERMINED"], total),
        }
        for field, counts in unknown_counts.items()
    }

    return {
        "statistics_schema_version": "0.1",
        "total_records": total,
        "distributions": {
            "source": _sorted_counts(sources),
            "language": _sorted_counts(languages),
            "domain": _sorted_counts(domains),
            "attack_family": _sorted_counts(attack_families),
            "attack_objective": _sorted_counts(attack_objectives),
            "delivery_mode": _sorted_counts(delivery_modes),
            "annotation_tier": _sorted_counts(tiers),
        },
        "mapping_quality": {
            "tiers": tier_metrics,
            "deterministic_mapping": tier_metrics["DETERMINISTIC_MAPPING"],
            "heuristic_mapping": tier_metrics["HEURISTIC_MAPPING"],
            "requires_manual_review": _metric(manual_review_count, total),
            "review_candidates": _metric(review_candidate_count, total),
            "mapping_confidence": _numeric_summary(mapping_confidences),
            "missing_mapping_confidence": _metric(missing_mapping_confidence, total),
        },
        "unknown_or_undetermined": unknown_metrics,
        "content_length": {
            "characters": _numeric_summary(character_lengths),
            "utf8_bytes": _numeric_summary(utf8_lengths),
        },
        "spans": {
            "total": len(span_lengths),
            "records_with_spans": _metric(records_with_spans, total),
            "count_per_record": _numeric_summary(span_counts),
            "length_characters": _numeric_summary(span_lengths),
            "by_type": _sorted_counts(span_types),
        },
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _percentage(value: Any) -> str:
    if not isinstance(value, Real) or isinstance(value, bool):
        return "0.00%"
    return f"{float(value) * 100:.2f}%"


def _distribution_table(counts: Mapping[str, int], total: int) -> list[str]:
    lines = ["| Value | Records | Share |", "|---|---:|---:|"]
    if not counts:
        lines.append("| - | 0 | 0.00% |")
        return lines
    for label in sorted(counts):
        count = counts[label]
        lines.append(f"| {_markdown_cell(label)} | {count} | {_percentage(_rate(count, total))} |")
    return lines


def _summary_table(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    lines = [
        "| Measure | Count | Min | Mean | Median | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(name),
                    str(summary.get("count", 0)),
                    str(summary.get("min", 0)),
                    str(summary.get("mean", 0)),
                    str(summary.get("median", 0)),
                    str(summary.get("p95", 0)),
                    str(summary.get("max", 0)),
                ]
            )
            + " |"
        )
    return lines


def render_statistics_markdown(stats: Mapping[str, Any]) -> str:
    """Render ``compute_statistics`` output without dates or unstable ordering."""

    total_value = stats.get("total_records", 0)
    total = total_value if isinstance(total_value, int) and not isinstance(total_value, bool) else 0
    distributions = stats.get("distributions")
    distributions = distributions if isinstance(distributions, Mapping) else {}
    mapping_quality = stats.get("mapping_quality")
    mapping_quality = mapping_quality if isinstance(mapping_quality, Mapping) else {}
    unknown = stats.get("unknown_or_undetermined")
    unknown = unknown if isinstance(unknown, Mapping) else {}
    content_length = stats.get("content_length")
    content_length = content_length if isinstance(content_length, Mapping) else {}
    spans = stats.get("spans")
    spans = spans if isinstance(spans, Mapping) else {}
    release = stats.get("release")
    release = release if isinstance(release, Mapping) else {}

    lines = [
        "# PromptSec-Dataset v0.1 statistics",
        "",
        "This report is generated deterministically from canonical records. P95 uses the "
        "nearest-rank method.",
        "",
        "## Overview",
        "",
        f"- Total records: **{total}**",
        f"- Total spans: **{spans.get('total', 0)}**",
    ]
    if release:
        lines.extend(
            [
                f"- Materialized after exact deduplication: "
                f"**{release.get('released_records', 0)}**",
                f"- Dropped exact duplicates: **{release.get('dropped_exact_duplicates', 0)}**",
                f"- Semantic clusters: **{release.get('semantic_clusters', 0)}**",
                f"- Review queue: **{release.get('review_queue_records', 0)}**",
            ]
        )
    lines.extend(["", "## Distributions"])

    distribution_titles = (
        ("source", "Source"),
        ("language", "Language"),
        ("domain", "Domain"),
        ("attack_family", "Attack family"),
        ("attack_objective", "Attack objective"),
        ("delivery_mode", "Delivery mode"),
        ("annotation_tier", "Annotation tier"),
    )
    for key, title in distribution_titles:
        counts = distributions.get(key)
        counts = counts if isinstance(counts, Mapping) else {}
        lines.extend(["", f"### {title}", "", *_distribution_table(counts, total)])

    lines.extend(
        [
            "",
            "## Mapping quality",
            "",
            "| Indicator | Records | Rate |",
            "|---|---:|---:|",
        ]
    )
    tiers = mapping_quality.get("tiers")
    tiers = tiers if isinstance(tiers, Mapping) else {}
    for tier, label in (
        ("GOLD_SOURCE", "Gold source"),
        ("DETERMINISTIC_MAPPING", "Deterministic mapping"),
        ("HEURISTIC_MAPPING", "Heuristic mapping"),
        ("UNANNOTATED", "Unannotated"),
    ):
        metric = tiers.get(tier)
        metric = metric if isinstance(metric, Mapping) else {}
        lines.append(
            f"| {label} | {metric.get('count', 0)} | {_percentage(metric.get('rate', 0.0))} |"
        )
    for key, label in (
        ("requires_manual_review", "Requires manual review"),
        ("review_candidates", "Review candidates"),
        ("review_queue", "Review queue after threshold"),
        ("missing_mapping_confidence", "Missing mapping confidence"),
    ):
        metric = mapping_quality.get(key)
        metric = metric if isinstance(metric, Mapping) else {}
        lines.append(
            f"| {label} | {metric.get('count', 0)} | {_percentage(metric.get('rate', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## UNKNOWN and UNDETERMINED fields",
            "",
            "| Field | UNKNOWN | UNDETERMINED | Combined | Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field in sorted(unknown):
        metric = unknown[field]
        metric = metric if isinstance(metric, Mapping) else {}
        combined = metric.get("combined")
        combined = combined if isinstance(combined, Mapping) else {}
        lines.append(
            f"| {_markdown_cell(field)} | {metric.get('UNKNOWN', 0)} | "
            f"{metric.get('UNDETERMINED', 0)} | {combined.get('count', 0)} | "
            f"{_percentage(combined.get('rate', 0.0))} |"
        )

    character_summary = content_length.get("characters")
    character_summary = character_summary if isinstance(character_summary, Mapping) else {}
    byte_summary = content_length.get("utf8_bytes")
    byte_summary = byte_summary if isinstance(byte_summary, Mapping) else {}
    span_count_summary = spans.get("count_per_record")
    span_count_summary = span_count_summary if isinstance(span_count_summary, Mapping) else {}
    span_length_summary = spans.get("length_characters")
    span_length_summary = span_length_summary if isinstance(span_length_summary, Mapping) else {}
    lines.extend(
        [
            "",
            "## Lengths",
            "",
            *_summary_table(
                (
                    ("Content characters", character_summary),
                    ("Content UTF-8 bytes", byte_summary),
                    ("Spans per record", span_count_summary),
                    ("Span length in characters", span_length_summary),
                )
            ),
            "",
            "## Span types",
            "",
        ]
    )
    by_type = spans.get("by_type")
    by_type = by_type if isinstance(by_type, Mapping) else {}
    lines.extend(_distribution_table(by_type, spans.get("total", 0)))
    return "\n".join(lines) + "\n"


__all__ = ["compute_statistics", "render_statistics_markdown"]
