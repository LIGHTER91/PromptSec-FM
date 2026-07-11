"""Deterministic construction of the audited PromptSec-Dataset v0.1 release."""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptsec.data.config import SourceConfig
from promptsec.data.fetch import fetch_artifacts
from promptsec.data.hashing import sha256_file
from promptsec.data.importers.base import load_importer
from promptsec.data.quality.deduplication import DedupConfig, DedupResult, analyze_duplicates
from promptsec.data.quality.grouping import assign_group
from promptsec.data.quality.mapping import assess_mapping, build_review_queue
from promptsec.data.quality.splitting import SplitConfig, SplitResult, assign_splits
from promptsec.data.quality.statistics import compute_statistics, render_statistics_markdown
from promptsec.data.release_config import DatasetReleaseConfig
from promptsec.data.validation import require_valid_record

_SPLIT_NAMES = (
    "train",
    "validation",
    "test_id",
    "test_held_out_source",
    "test_held_out_family",
)
_HASH_FIELDS = ("raw_hash", "normalized_hash", "contextual_hash")


class ReleaseBuildError(RuntimeError):
    """Raised when a release cannot be assembled without violating its contract."""


@dataclass(frozen=True, slots=True)
class ReleaseBuildReport:
    release_id: str
    output: str
    imported_records: int
    released_records: int
    review_queue_records: int
    split_records: dict[str, int]
    dropped_exact_duplicates: int
    semantic_clusters: int
    publication_status: str
    checksums_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseAnalysis:
    records: tuple[dict[str, Any], ...]
    records_by_id: dict[str, dict[str, Any]]
    review_queue: tuple[dict[str, Any], ...]
    statistics: dict[str, Any]
    deduplication: DedupResult
    splits: SplitResult


def build_release(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> ReleaseBuildReport:
    """Fetch pinned sources and build the complete audited release."""

    config = DatasetReleaseConfig.load(config_path)
    output = _resolve_output(config, output_override)
    records, sources, licenses = _import_sources(config)
    analysis = analyze_release_records(records, config)
    _write_release(output, config, analysis, sources, licenses)
    _write_external_reports(config, analysis)

    split_records = {name: len(analysis.splits.splits[name]) for name in _SPLIT_NAMES}
    summary = analysis.deduplication.report["summary"]
    return ReleaseBuildReport(
        release_id=config.identity.id,
        output=_portable_path(output, config.project_root),
        imported_records=len(analysis.records),
        released_records=sum(split_records.values()),
        review_queue_records=len(analysis.review_queue),
        split_records=split_records,
        dropped_exact_duplicates=int(summary["dropped_exact_duplicates"]),
        semantic_clusters=int(summary["semantic_clusters"]),
        publication_status=licenses["publication_status"],
        checksums_sha256=sha256_file(output / "checksums.sha256"),
    )


def analyze_release_records(
    records: list[dict[str, Any]], config: DatasetReleaseConfig
) -> ReleaseAnalysis:
    """Attach auditable quality fields and produce leakage-safe split assignments."""

    materialized = [copy.deepcopy(record) for record in records]
    ids = [record.get("id") for record in materialized]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ReleaseBuildError("every imported record must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ReleaseBuildError("imported records contain duplicate canonical ids")
    ordered = sorted(materialized, key=lambda item: item["id"])

    mappings: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for record in ordered:
        require_valid_record(record)
        record_id = record["id"]
        mappings[record_id] = assess_mapping(record, config.mapping_quality.profiles)
        groups[record_id] = assign_group(record)

    semantic_group_keys = {
        record["id"]: group_key
        for record in ordered
        if (group_key := _source_semantic_group_key(record, groups[record["id"]])) is not None
    }

    deduplication = analyze_duplicates(
        ordered,
        DedupConfig(
            semantic_threshold=config.deduplication.semantic_threshold,
            variant_threshold=config.deduplication.variant_threshold,
        ),
        semantic_group_keys=semantic_group_keys,
    )
    splits = assign_splits(
        ordered,
        deduplication.assignments,
        groups,
        SplitConfig(
            seed=config.identity.seed,
            held_out_source=config.splits.held_out_source,
            held_out_family=config.splits.held_out_family,
            general_ratios=config.splits.general_ratios,
            notinject_ratios=config.splits.notinject_ratios,
        ),
    )
    if not splits.report["constraints"]["all_satisfied"]:
        raise ReleaseBuildError("split construction violated one or more leakage constraints")

    enriched: list[dict[str, Any]] = []
    release_schema = config.project_root / "schemas" / "promptsec-release-record-v0.1.schema.json"
    if not release_schema.is_file():
        # Synthetic tests may place their release config outside the checkout.
        release_schema = Path(__file__).resolve().parents[3] / "schemas" / release_schema.name
    for record in ordered:
        record_id = record["id"]
        dedup_assignment = copy.deepcopy(deduplication.assignments[record_id])
        hashes = {field: dedup_assignment[field] for field in _HASH_FIELDS}
        dedup_fields = {
            key: value for key, value in dedup_assignment.items() if key not in _HASH_FIELDS
        }
        extensions = record.setdefault("extensions", {})
        extensions["quality_v0_1"] = {
            "quality_schema_version": "0.1",
            "mapping_quality": copy.deepcopy(mappings[record_id]),
            "grouping": copy.deepcopy(groups[record_id]),
            "hashes": hashes,
            "deduplication": dedup_fields,
            "split": splits.assignments[record_id],
        }
        require_valid_record(record, schema_path=release_schema)
        enriched.append(record)

    statistics = compute_statistics(
        enriched, review_threshold=config.mapping_quality.review_threshold
    )
    statistics["release"] = {
        "id": config.identity.id,
        "taxonomy_version": config.identity.taxonomy_version,
        "imported_records": len(enriched),
        "released_records": sum(len(values) for values in splits.splits.values()),
        "dropped_exact_duplicates": len(splits.report["dropped_exact_ids"]),
        "semantic_clusters": deduplication.report["summary"]["semantic_clusters"],
        "records_by_split": {name: len(splits.splits[name]) for name in _SPLIT_NAMES},
    }
    review_queue = build_review_queue(enriched, config.mapping_quality.review_threshold)
    statistics["release"]["review_queue_records"] = len(review_queue)
    statistics["mapping_quality"]["review_queue"] = {
        "count": len(review_queue),
        "rate": round(len(review_queue) / len(enriched), 6) if enriched else 0.0,
        "threshold": config.mapping_quality.review_threshold,
    }
    records_by_id = {record["id"]: record for record in enriched}
    return ReleaseAnalysis(
        records=tuple(enriched),
        records_by_id=records_by_id,
        review_queue=tuple(review_queue),
        statistics=statistics,
        deduplication=deduplication,
        splits=splits,
    )


def verify_release_checksums(output: str | Path) -> list[str]:
    """Return checksum failures; an empty list means the release is intact."""

    root = Path(output).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        return ["missing checksums.sha256"]
    failures: list[str] = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"checksums.sha256:{line_number}: malformed line")
            continue
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            failures.append(f"checksums.sha256:{line_number}: unsafe path {relative}")
            continue
        if not candidate.is_file():
            failures.append(f"missing {relative}")
        else:
            actual = sha256_file(candidate)
            if actual != expected:
                failures.append(f"checksum mismatch for {relative}: {actual} != {expected}")
    return failures


def _import_sources(
    config: DatasetReleaseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    license_entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    for source_path in config.source_configs:
        source = SourceConfig.load(source_path)
        if source.id in seen_sources:
            raise ReleaseBuildError(f"duplicate configured source id: {source.id}")
        if source.id not in config.mapping_quality.profiles:
            raise ReleaseBuildError(f"missing mapping-quality profile for source {source.id}")
        seen_sources.add(source.id)

        artifact_paths = fetch_artifacts(source, config.paths.raw_dir)
        importer = load_importer(
            source.importer,
            source,
            imported_at=config.identity.imported_at,
        )
        source_records = list(importer.records(artifact_paths))
        for record in source_records:
            _normalize_record_paths(record, source, config.project_root)
            require_valid_record(record)
        records.extend(source_records)

        artifact_entries = []
        for artifact in source.artifacts:
            local_path = artifact_paths[artifact.id]
            artifact_entries.append(
                {
                    "id": artifact.id,
                    "split": artifact.split,
                    "format": artifact.format,
                    "url": artifact.url,
                    "expected_sha256": artifact.sha256,
                    "observed_sha256": sha256_file(local_path),
                    "local_path": _portable_path(local_path, config.project_root),
                    "records_path": artifact.records_path,
                }
            )
        source_entries.append(
            {
                "id": source.id,
                "name": source.name,
                "homepage": source.homepage,
                "repository": source.repository,
                "version": source.version,
                "revision": source.revision,
                "records": len(source_records),
                "config": _portable_path(source.path, config.project_root),
                "config_sha256": source.sha256,
                "license_manifest": source.license_manifest,
                "artifacts": artifact_entries,
            }
        )

        manifest_path = (config.project_root / source.license_manifest).resolve()
        manifest = _read_json_object(manifest_path)
        if manifest.get("source_id") != source.id:
            raise ReleaseBuildError(
                f"license manifest {manifest_path} does not match source {source.id}"
            )
        license_entries.append(
            {
                "manifest_path": _portable_path(manifest_path, config.project_root),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest": manifest,
            }
        )

    if set(config.mapping_quality.profiles) != seen_sources:
        extras = sorted(set(config.mapping_quality.profiles) - seen_sources)
        raise ReleaseBuildError(f"mapping-quality profiles have no configured source: {extras}")

    sources = {
        "schema_version": "0.1",
        "release_id": config.identity.id,
        "release_config": _portable_path(config.path, config.project_root),
        "release_config_sha256": config.sha256,
        "sources": sorted(source_entries, key=lambda item: item["id"]),
    }
    licenses = {
        "schema_version": "0.1",
        "release_id": config.identity.id,
        "notice": (
            "License obligations remain component-specific; NOASSERTION and conditional "
            "entries require release-time review."
        ),
        "sources": sorted(license_entries, key=lambda item: item["manifest"]["source_id"]),
    }
    publication_blockers = [
        {
            "source_id": entry["manifest"]["source_id"],
            "scope": component.get("scope"),
            "license_expression": component.get("license_expression"),
            "redistribution": component.get("redistribution"),
            "reason": "Redistribution permission is unresolved in the source manifest.",
        }
        for entry in licenses["sources"]
        for component in entry["manifest"].get("components", [])
        if component.get("redistribution") == "unknown"
    ]
    licenses["publication_status"] = (
        "BLOCKED_PENDING_LICENSE_REVIEW" if publication_blockers else "MANIFEST_REVIEW_COMPLETE"
    )
    licenses["publication_blockers"] = publication_blockers
    return records, sources, licenses


def _write_release(
    output: Path,
    config: DatasetReleaseConfig,
    analysis: ReleaseAnalysis,
    sources: dict[str, Any],
    licenses: dict[str, Any],
) -> None:
    staging = output.with_name(f".{output.name}.tmp")
    _reset_directory(staging)
    try:
        for split_name in _SPLIT_NAMES:
            split_records = [
                analysis.records_by_id[record_id]
                for record_id in analysis.splits.splits[split_name]
            ]
            _write_jsonl(staging / f"{split_name}.jsonl", split_records)

        _write_jsonl(staging / "review_queue.jsonl", analysis.review_queue)
        _write_json(staging / "statistics.json", analysis.statistics)
        _write_json(staging / "sources.json", sources)
        _write_json(staging / "licenses.json", licenses)

        deduplication_report = copy.deepcopy(analysis.deduplication.report)
        deduplication_report["release_id"] = config.identity.id
        deduplication_report["record_assignments"] = copy.deepcopy(
            analysis.deduplication.assignments
        )
        _write_json(staging / "deduplication_report.json", deduplication_report)

        split_report = copy.deepcopy(analysis.splits.report)
        split_report["release_id"] = config.identity.id
        split_report["record_assignments"] = copy.deepcopy(analysis.splits.assignments)
        _write_json(staging / "split_report.json", split_report)
        (staging / "dataset_card.md").write_text(
            _dataset_card(config, analysis, licenses), encoding="utf-8", newline="\n"
        )
        _write_checksums(staging)
        failures = verify_release_checksums(staging)
        if failures:
            raise ReleaseBuildError("release checksum verification failed: " + "; ".join(failures))

        if output.exists():
            _safe_remove_directory(output)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            _safe_remove_directory(staging)
        raise


def _write_external_reports(config: DatasetReleaseConfig, analysis: ReleaseAnalysis) -> None:
    _write_json(config.paths.statistics_json, analysis.statistics)
    config.paths.statistics_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.paths.statistics_markdown.write_text(
        render_statistics_markdown(analysis.statistics),
        encoding="utf-8",
        newline="\n",
    )
    _write_jsonl(config.paths.review_queue, analysis.review_queue)


def _dataset_card(
    config: DatasetReleaseConfig,
    analysis: ReleaseAnalysis,
    licenses: dict[str, Any],
) -> str:
    stats = analysis.statistics
    source_counts = stats["distributions"]["source"]
    tier_counts = stats["mapping_quality"]["tiers"]
    split_counts = stats["release"]["records_by_split"]
    constraints = analysis.splits.report["constraints"]
    lines = [
        f"# {config.identity.title}",
        "",
        "PromptSec-Dataset v0.1 is an **experimental audited release**. It is not a "
        "final training split, and this build does not train a model.",
        "",
        "## Scope",
        "",
        f"- Frozen normative taxonomy: PromptSec-FM v{config.identity.taxonomy_version}",
        f"- Imported source records: {stats['total_records']}",
        f"- Materialized records after exact deduplication: {stats['release']['released_records']}",
        f"- Semantic clusters: {stats['release']['semantic_clusters']}",
        f"- Review-queue records: {stats['release']['review_queue_records']}",
        "",
        "## Source distribution before deduplication",
        "",
        "| Source | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| {source} | {count} |" for source, count in sorted(source_counts.items()))
    lines.extend(
        [
            "",
            "## Mapping evidence",
            "",
            "Mapping tiers describe the evidence used to migrate source labels. They must not "
            "be treated as equivalent to human annotation.",
            "",
            "| Tier | Records |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {tier} | {metric['count']} |" for tier, metric in sorted(tier_counts.items()))
    lines.extend(
        [
            "",
            "## Experimental splits",
            "",
            "| Split | Records |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {name} | {split_counts[name]} |" for name in _SPLIT_NAMES)
    lines.extend(
        [
            "",
            "NotInject is weighted toward validation and `test_id` to measure over-defense. "
            f"Source `{config.splits.held_out_source}` is reserved for "
            "`test_held_out_source`; family "
            f"`{config.splits.held_out_family}` is reserved for `test_held_out_family`.",
            "",
            "Leakage checks:",
            "",
            "- No semantic cluster crosses materialized splits: "
            f"{constraints['no_cluster_leakage']}",
            f"- Held-out source absent from train: {constraints['no_held_out_source_in_train']}",
            f"- Held-out family absent from train: {constraints['no_held_out_family_in_train']}",
            "- No template family overlaps held-out-family test and train: "
            f"{constraints['no_template_family_overlap_with_train']}",
            f"- Exact duplicates excluded: {constraints['exact_duplicates_excluded']}",
            "",
            "## Deduplication and grouping",
            "",
            "Exact grouping uses raw, normalized, and context-aware SHA-256 hashes. Semantic "
            "clusters use a deterministic lexical similarity rule with documented synonym "
            "normalization; no learned embedding or model is used. Paraphrases are retained as "
            "variants or sent to review, and clusters remain atomic across splits.",
            "",
            "Template families come only from source metadata, generation templates, documented "
            "mapping rules, or manual review. See `docs/family_mapping_v0.1.md`.",
            "",
            "## Limitations and licensing",
            "",
            "The 627 imported records are strongly source-imbalanced. PromptInject and "
            "Open-Prompt-Injection are small, while BIPIA and NotInject dominate. The release "
            "contains uncertain and non-annotated mappings explicitly isolated in the review "
            "queue. Consult `licenses.json` before redistribution: obligations are component-"
            "specific.",
            "",
            f"Publication status: **{licenses['publication_status']}**. BIPIA attack-template "
            "redistribution is recorded as NOASSERTION/unknown, so this local experimental "
            "artifact must not be published until that license review is resolved.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/build_dataset.py --config configs/dataset_v0.1.yaml "
            "--output data/releases/promptsec-dataset-v0.1",
            "```",
            "",
            "`checksums.sha256` covers every other release file. With the pinned artifacts and "
            "configuration, two builds are byte-identical.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"expected a JSON object in {path}")
    return value


def _normalize_record_paths(
    record: dict[str, Any], source: SourceConfig, project_root: Path
) -> None:
    provenance = record["metadata"]["dataset_provenance"]
    raw_artifact = provenance["source_record"]["raw_artifact"]
    raw_path = Path(raw_artifact)
    if raw_path.is_absolute():
        provenance["source_record"]["raw_artifact"] = _portable_path(raw_path, project_root)
    else:
        provenance["source_record"]["raw_artifact"] = raw_path.as_posix()
    provenance["import"]["config"] = _portable_path(source.path, project_root)


def _source_semantic_group_key(record: dict[str, Any], grouping: dict[str, Any]) -> str | None:
    """Return a structured source-template key without inspecting prompt text."""

    provenance = record["metadata"]["dataset_provenance"]
    source_id = provenance["source_dataset"]["id"]
    original_fields = provenance["source_record"]["original_fields"]
    if source_id == "bipia":
        payload_type = original_fields.get("payload_type")
        domain = grouping.get("domain")
        if isinstance(payload_type, str) and payload_type and isinstance(domain, str) and domain:
            return f"bipia:{payload_type}:{domain}"
    if source_id == "open_prompt_injection":
        task_config = original_fields.get("task_config")
        if isinstance(task_config, str) and task_config:
            return f"open_prompt_injection:{task_config}"
    return None


def _resolve_output(config: DatasetReleaseConfig, output_override: str | Path | None) -> Path:
    if output_override is None:
        resolved = config.paths.output.resolve()
    else:
        candidate = Path(output_override)
        resolved = (
            (config.project_root / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
    releases_root = (config.project_root / "data" / "releases").resolve()
    if resolved == releases_root or not resolved.is_relative_to(releases_root):
        raise ReleaseBuildError(
            f"release output must be a child of {releases_root}, got {resolved}"
        )
    return resolved


def _portable_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _reset_directory(path: Path) -> None:
    if path.exists():
        _safe_remove_directory(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _safe_remove_directory(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor).resolve()
    if resolved == anchor or resolved == resolved.parent:
        raise ReleaseBuildError(f"refusing to remove unsafe directory: {resolved}")
    if not resolved.is_dir():
        raise ReleaseBuildError(f"expected a directory: {resolved}")
    shutil.rmtree(resolved)


__all__ = [
    "ReleaseAnalysis",
    "ReleaseBuildError",
    "ReleaseBuildReport",
    "analyze_release_records",
    "build_release",
    "verify_release_checksums",
]
