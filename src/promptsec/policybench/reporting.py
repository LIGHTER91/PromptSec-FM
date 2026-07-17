"""Text-free aggregate quality reporting for PromptSec-PolicyBench."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from promptsec.policybench.deduplication import (
    DuplicateAnalysis,
    analyze_policybench_duplicates,
)
from promptsec.policybench.io import write_json, write_named_checksums


class PolicyBenchReportingError(ValueError):
    """Raised when a quality report cannot be computed safely."""


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(value, Mapping):
        raise PolicyBenchReportingError(
            f"record {record.get('id')!r}: extensions.policybench_v0_1 must be an object"
        )
    for field in ("policy", "blueprint", "generation", "validation", "grouping"):
        if not isinstance(value.get(field), Mapping):
            raise PolicyBenchReportingError(
                f"record {record.get('id')!r}: policybench extension {field} must be an object"
            )
    return value


def _increment(counter: Counter[str], value: Any) -> None:
    counter[str(value) if isinstance(value, str) and value else "UNKNOWN"] += 1


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    return {name: counter[name] for name in sorted(counter)}


def _increment_multilabel(counter: Counter[str], value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise PolicyBenchReportingError(f"{context} must be an array")
    if not value:
        counter["NONE"] += 1
        return
    labels: list[str] = []
    for label in value:
        if not isinstance(label, str) or not label:
            raise PolicyBenchReportingError(f"{context} must contain non-empty strings")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise PolicyBenchReportingError(f"{context} must not contain duplicate labels")
    counter.update(labels)


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    if percentile <= 0:
        return values[0]
    rank = math.ceil(percentile / 100 * len(values))
    return values[min(len(values) - 1, max(0, rank - 1))]


def _length_statistics(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    percentiles = (0, 25, 50, 75, 90, 95, 99, 100)
    return {
        "unit": "Python Unicode code points",
        "percentile_method": "nearest-rank",
        "count": len(ordered),
        "minimum": ordered[0] if ordered else 0,
        "maximum": ordered[-1] if ordered else 0,
        "mean": round(sum(ordered) / len(ordered), 6) if ordered else 0.0,
        "percentiles": {
            f"p{percentile}": _nearest_rank(ordered, percentile) for percentile in percentiles
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _duplicate_summary(value: DuplicateAnalysis | Mapping[str, Any]) -> dict[str, Any]:
    report = value.report if isinstance(value, DuplicateAnalysis) else value
    if not isinstance(report, Mapping):
        raise PolicyBenchReportingError("duplicate analysis report must be an object")
    summary = report.get("summary")
    decisions = report.get("decision_counts")
    if not isinstance(summary, Mapping) or not isinstance(decisions, Mapping):
        raise PolicyBenchReportingError("duplicate report is missing summary or decisions")
    return {
        "algorithm": report.get("algorithm"),
        "summary": dict(summary),
        "decision_counts": dict(decisions),
    }


def build_quality_report(
    records: Iterable[Mapping[str, Any]],
    duplicate_analysis: DuplicateAnalysis | Mapping[str, Any] | None = None,
    *,
    semantic_duplicate_threshold: float = 0.93,
    split_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic distributions, failures, retries, lengths, and costs."""

    materialized = list(records)
    ids: set[str] = set()
    distributions: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: list[int] = []
    retry_reasons: Counter[str] = Counter()
    attempt_counts: Counter[str] = Counter()
    failed_checks: Counter[str] = Counter()
    span_failure_records = 0
    span_failure_checks = 0
    span_rejected_attempts = 0
    records_with_retries = 0
    total_failed_attempts = 0
    total_generation_attempts = 0
    automatic_gold_records = 0
    token_totals = Counter()
    known_cost = Decimal("0")
    records_with_usage = 0
    records_with_known_cost = 0
    unknown_cost_records = 0
    cost_by_generator: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    policy_ids: set[str] = set()
    policy_hashes: set[str] = set()
    counterfactual_type_by_group: dict[str, str] = {}
    counterfactual_group_sizes: Counter[str] = Counter()

    for record in materialized:
        if not isinstance(record, Mapping):
            raise PolicyBenchReportingError("every record must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise PolicyBenchReportingError("every record must have a non-empty string id")
        if record_id in ids:
            raise PolicyBenchReportingError(f"duplicate canonical record id: {record_id}")
        ids.add(record_id)
        extension = _extension(record)
        policy = extension["policy"]
        blueprint = extension["blueprint"]
        generation = extension["generation"]
        validation = extension["validation"]
        assert isinstance(policy, Mapping)
        assert isinstance(blueprint, Mapping)
        assert isinstance(generation, Mapping)
        assert isinstance(validation, Mapping)
        content = record.get("content")
        context = record.get("context")
        annotations = record.get("annotations")
        derived = record.get("derived")
        if (
            not isinstance(content, Mapping)
            or not isinstance(context, Mapping)
            or not isinstance(annotations, Mapping)
        ):
            raise PolicyBenchReportingError(
                f"record {record_id!r}: context, content, and annotations must be objects"
            )
        derived = derived if isinstance(derived, Mapping) else {}
        text = content.get("text")
        if not isinstance(text, str):
            raise PolicyBenchReportingError(f"record {record_id!r}: content.text must be a string")
        lengths.append(len(text))

        for name, value in (
            ("domain", blueprint.get("domain")),
            ("language", blueprint.get("language")),
            ("category", blueprint.get("category")),
            ("policy_family", policy.get("policy_family")),
            ("data_quality", extension.get("data_quality")),
            ("human_validation_status", extension.get("human_validation_status")),
            ("generation_method", extension.get("generation_method")),
            ("dataset_split", extension.get("dataset_split")),
            ("generator_provider", generation.get("generator_provider")),
            ("generator_model", generation.get("generator_model")),
            ("validation_status", validation.get("overall_status")),
            ("source_role", content.get("source_role")),
            ("instruction_presence", annotations.get("instruction_presence")),
            ("instruction_presentation", annotations.get("instruction_presentation")),
            ("user_goal_alignment", annotations.get("user_goal_alignment")),
            (
                "protected_policy_alignment",
                annotations.get("protected_policy_alignment"),
            ),
            ("authority_status", annotations.get("authority_status")),
            ("prompt_injection_verdict", derived.get("prompt_injection_verdict")),
        ):
            _increment(distributions[name], value)

        protected_policy = context.get("protected_policy")
        if protected_policy is None:
            protected_policy_presence = "MISSING"
        elif isinstance(protected_policy, str) and not protected_policy:
            protected_policy_presence = "EMPTY"
        elif isinstance(protected_policy, str):
            protected_policy_presence = "PRESENT"
        else:
            raise PolicyBenchReportingError(
                f"record {record_id!r}: context.protected_policy must be a string or null"
            )
        distributions["protected_policy_presence"][protected_policy_presence] += 1
        policy_id = policy.get("policy_id")
        policy_hash = policy.get("policy_sha256")
        if not isinstance(policy_id, str) or not policy_id:
            raise PolicyBenchReportingError(f"record {record_id!r}: policy_id must be a string")
        if not isinstance(policy_hash, str) or not policy_hash:
            raise PolicyBenchReportingError(f"record {record_id!r}: policy_sha256 must be a string")
        distributions["policy_id"][policy_id] += 1
        policy_ids.add(policy_id)
        policy_hashes.add(policy_hash)
        _increment_multilabel(
            distributions["attack_families"],
            annotations.get("attack_families"),
            f"record {record_id!r} annotations.attack_families",
        )
        _increment_multilabel(
            distributions["attack_objectives"],
            annotations.get("attack_objectives"),
            f"record {record_id!r} annotations.attack_objectives",
        )

        counterfactual = extension.get("counterfactual")
        if counterfactual is None:
            distributions["counterfactual_type"]["NONE"] += 1
        elif isinstance(counterfactual, Mapping):
            group_id = counterfactual.get("counterfactual_group_id")
            counterfactual_type = counterfactual.get("counterfactual_type")
            if not isinstance(group_id, str) or not group_id:
                raise PolicyBenchReportingError(
                    f"record {record_id!r}: counterfactual_group_id must be a string"
                )
            if not isinstance(counterfactual_type, str) or not counterfactual_type:
                raise PolicyBenchReportingError(
                    f"record {record_id!r}: counterfactual_type must be a string"
                )
            previous_type = counterfactual_type_by_group.setdefault(group_id, counterfactual_type)
            if previous_type != counterfactual_type:
                raise PolicyBenchReportingError(
                    f"counterfactual group {group_id!r} declares conflicting types"
                )
            counterfactual_group_sizes[group_id] += 1
            distributions["counterfactual_type"][counterfactual_type] += 1
        else:
            raise PolicyBenchReportingError(
                f"record {record_id!r}: counterfactual must be an object or null"
            )

        data_quality = extension.get("data_quality")
        human_status = extension.get("human_validation_status")
        if data_quality == "GOLD_HUMAN_CONFIRMED" or human_status != "PENDING":
            automatic_gold_records += 1

        generation_attempt = generation.get("generation_attempt")
        if (
            isinstance(generation_attempt, bool)
            or not isinstance(generation_attempt, int)
            or generation_attempt < 1
        ):
            raise PolicyBenchReportingError(
                f"record {record_id!r}: generation_attempt must be a positive integer"
            )
        failed_attempts = generation.get("failed_attempts")
        if not isinstance(failed_attempts, list):
            raise PolicyBenchReportingError(
                f"record {record_id!r}: failed_attempts must be an array"
            )
        total_generation_attempts += generation_attempt
        total_failed_attempts += len(failed_attempts)
        attempt_counts[str(generation_attempt)] += 1
        if generation_attempt > 1 or failed_attempts:
            records_with_retries += 1
        for failed_attempt in failed_attempts:
            if not isinstance(failed_attempt, Mapping):
                raise PolicyBenchReportingError(
                    f"record {record_id!r}: failed attempt must be an object"
                )
            reasons = failed_attempt.get("rejection_reasons")
            if isinstance(reasons, list):
                if any("span" in str(reason).casefold() for reason in reasons):
                    span_rejected_attempts += 1
                for reason in reasons:
                    _increment(retry_reasons, reason)

        checks = validation.get("checks")
        if not isinstance(checks, list):
            raise PolicyBenchReportingError(
                f"record {record_id!r}: validation.checks must be an array"
            )
        record_has_span_failure = False
        for check in checks:
            if not isinstance(check, Mapping):
                raise PolicyBenchReportingError(
                    f"record {record_id!r}: validation check must be an object"
                )
            if check.get("status") != "FAIL":
                continue
            check_name = str(check.get("name", "UNKNOWN"))
            failed_checks[check_name] += 1
            if check_name == "SPANS":
                span_failure_checks += 1
                record_has_span_failure = True
        span_failure_records += int(record_has_span_failure)

        usage = generation.get("usage")
        if usage is None:
            unknown_cost_records += 1
            continue
        if not isinstance(usage, Mapping):
            raise PolicyBenchReportingError(
                f"record {record_id!r}: generation.usage must be an object"
            )
        records_with_usage += 1
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            token_value = usage.get(name)
            if isinstance(token_value, bool) or not isinstance(token_value, int):
                raise PolicyBenchReportingError(
                    f"record {record_id!r}: usage.{name} must be an integer"
                )
            token_totals[name] += token_value
        cost = usage.get("cost_usd")
        if cost is None:
            unknown_cost_records += 1
            continue
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise PolicyBenchReportingError(
                f"record {record_id!r}: usage.cost_usd must be non-negative or null"
            )
        decimal_cost = Decimal(str(cost))
        known_cost += decimal_cost
        records_with_known_cost += 1
        generator_key = (
            f"{generation.get('generator_provider', 'UNKNOWN')}/"
            f"{generation.get('generator_model', 'UNKNOWN')}"
        )
        cost_by_generator[generator_key] += decimal_cost

    if duplicate_analysis is None:
        duplicate_analysis = analyze_policybench_duplicates(
            materialized,
            semantic_threshold=semantic_duplicate_threshold,
            reject_exact=False,
            reject_normalized=False,
        )
    duplicate_summary = _duplicate_summary(duplicate_analysis)
    counterfactual_groups_by_type: Counter[str] = Counter(counterfactual_type_by_group.values())
    counterfactual_size_distribution = Counter(counterfactual_group_sizes.values())
    report = {
        "schema_version": "0.1",
        "phase_state": "SILVER_QUALITY_REVIEW",
        "records": len(materialized),
        "automatic_gold_records": automatic_gold_records,
        "gold_claim_permitted": False,
        "distributions": {
            name: _ordered_counts(counter) for name, counter in sorted(distributions.items())
        },
        "candidate_length": _length_statistics(lengths),
        "protected_policy_coverage": {
            "records_by_presence": _ordered_counts(distributions["protected_policy_presence"]),
            "distinct_policy_ids": len(policy_ids),
            "distinct_policy_hashes": len(policy_hashes),
        },
        "counterfactual_coverage": {
            "records": sum(counterfactual_group_sizes.values()),
            "groups": len(counterfactual_group_sizes),
            "records_by_type": _ordered_counts(distributions["counterfactual_type"]),
            "groups_by_type": _ordered_counts(counterfactual_groups_by_type),
            "group_size_distribution": {
                str(size): counterfactual_size_distribution[size]
                for size in sorted(counterfactual_size_distribution)
            },
        },
        "duplicates": duplicate_summary,
        "generation_retries": {
            "records_with_retries": records_with_retries,
            "retry_rate": _rate(records_with_retries, len(materialized)),
            "total_generation_attempts": total_generation_attempts,
            "total_failed_attempts": total_failed_attempts,
            "rejection_rate": _rate(total_failed_attempts, total_generation_attempts),
            "acceptance_rate": _rate(len(materialized), total_generation_attempts),
            "generation_attempt_distribution": _ordered_counts(attempt_counts),
            "failed_attempt_reasons": _ordered_counts(retry_reasons),
        },
        "validation_failures": {
            "failed_checks": _ordered_counts(failed_checks),
            "span_failure_records": span_failure_records,
            "span_failure_checks": span_failure_checks,
            "span_rejected_attempts": span_rejected_attempts,
            "span_rejection_rate": _rate(span_rejected_attempts, total_generation_attempts),
        },
        "generation_usage": {
            "records_with_usage": records_with_usage,
            "records_with_known_cost": records_with_known_cost,
            "unknown_cost_records": unknown_cost_records,
            "input_tokens": token_totals["input_tokens"],
            "output_tokens": token_totals["output_tokens"],
            "total_tokens": token_totals["total_tokens"],
            "known_cost_usd": float(known_cost),
            "known_cost_usd_by_generator": {
                name: float(cost_by_generator[name]) for name in sorted(cost_by_generator)
            },
        },
    }
    if split_report is not None:
        report["splits"] = {
            "records_by_split": split_report.get("records_by_split"),
            "constraints": split_report.get("constraints"),
            "component_diagnostics": split_report.get("component_diagnostics"),
        }
    return report


def render_quality_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise companion report without record text."""

    distributions = report.get("distributions")
    distributions = distributions if isinstance(distributions, Mapping) else {}
    lengths = report.get("candidate_length")
    lengths = lengths if isinstance(lengths, Mapping) else {}
    duplicates = report.get("duplicates")
    duplicates = duplicates if isinstance(duplicates, Mapping) else {}
    duplicate_summary = duplicates.get("summary")
    duplicate_summary = duplicate_summary if isinstance(duplicate_summary, Mapping) else {}
    policy_coverage = report.get("protected_policy_coverage")
    policy_coverage = policy_coverage if isinstance(policy_coverage, Mapping) else {}
    policy_presence = policy_coverage.get("records_by_presence")
    policy_presence = policy_presence if isinstance(policy_presence, Mapping) else {}
    counterfactual_coverage = report.get("counterfactual_coverage")
    counterfactual_coverage = (
        counterfactual_coverage if isinstance(counterfactual_coverage, Mapping) else {}
    )
    lines = [
        "# PromptSec-PolicyBench v0.1 quality report",
        "",
        f"Records: **{report.get('records', 0)}**",
        f"Automatic GOLD records: **{report.get('automatic_gold_records', 0)}**",
        "",
        "All records remain AI-generated SILVER data pending human validation.",
        "",
        "## Candidate length",
        "",
        f"Mean Unicode code points: **{lengths.get('mean', 0)}**",
        f"Maximum Unicode code points: **{lengths.get('maximum', 0)}**",
        "",
        "## Protected-policy coverage",
        "",
        f"Present policy context: **{policy_presence.get('PRESENT', 0)}**",
        f"Missing policy context: **{policy_presence.get('MISSING', 0)}**",
        f"Distinct policy IDs: **{policy_coverage.get('distinct_policy_ids', 0)}**",
        "",
        "## Counterfactual coverage",
        "",
        f"Counterfactual records: **{counterfactual_coverage.get('records', 0)}**",
        f"Counterfactual groups: **{counterfactual_coverage.get('groups', 0)}**",
        "",
        "## Duplicate analysis",
        "",
        f"Exact duplicate groups: **{duplicate_summary.get('exact_duplicate_groups', 0)}**",
        "Normalized duplicate groups: "
        f"**{duplicate_summary.get('normalized_duplicate_groups', 0)}**",
        "Semantic duplicate clusters: "
        f"**{duplicate_summary.get('semantic_duplicate_clusters', 0)}**",
        "",
        "## Distribution overview",
        "",
        "| Axis | Values |",
        "|---|---:|",
    ]
    for name, values in sorted(distributions.items()):
        lines.append(f"| {name} | {len(values) if isinstance(values, Mapping) else 0} |")
    return "\n".join(lines) + "\n"


def write_quality_report(report: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Write JSON/Markdown quality reports and a deterministic checksum list."""

    root = Path(output)
    write_json(root / "quality_report.json", report)
    markdown = render_quality_report_markdown(report)
    markdown_path = root / "quality_report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    names = ["quality_report.json", "quality_report.md"]
    write_named_checksums(root, names)
    return {
        "schema_version": "0.1",
        "output": root.as_posix(),
        "files": names,
        "checksums": "checksums.sha256",
    }


__all__ = [
    "PolicyBenchReportingError",
    "build_quality_report",
    "render_quality_report_markdown",
    "write_quality_report",
]
