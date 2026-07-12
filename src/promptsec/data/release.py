"""Deterministic construction of the audited PromptSec-Dataset v0.1 release."""

from __future__ import annotations

import copy
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptsec.data.acquisition import acquire_source
from promptsec.data.config import SourceConfig
from promptsec.data.hashing import sha256_file
from promptsec.data.importers.base import load_importer
from promptsec.data.quality.deduplication import DedupConfig, DedupResult, analyze_duplicates
from promptsec.data.quality.grouping import assign_group
from promptsec.data.quality.mapping import assess_mapping, build_review_queue
from promptsec.data.quality.splitting import SplitConfig, SplitResult, assign_splits
from promptsec.data.quality.statistics import compute_statistics, render_statistics_markdown
from promptsec.data.release_config import DatasetReleaseConfig
from promptsec.data.validation import require_valid_record

_HASH_FIELDS = ("raw_hash", "normalized_hash", "contextual_hash")
_FULL_CHECKSUMS = "checksums.sha256"
_PIPELINE_CHECKSUMS = "checksums_pipeline.txt"
_PUBLISHABLE_METADATA_FILES = (
    "dataset_card.md",
    "sources.json",
    "licenses.json",
    "statistics.json",
    "deduplication_report.json",
    "split_report.json",
    "release_manifest.json",
)


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
    offline: bool = False,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> ReleaseBuildReport:
    """Fetch pinned sources and build the complete audited release."""

    config = DatasetReleaseConfig.load(config_path)
    output = _resolve_output(config, output_override)
    records, sources, licenses = _import_sources(
        config,
        offline=offline,
        source_overrides=source_overrides or {},
    )
    analysis = analyze_release_records(records, config)
    _write_release(output, config, analysis, sources, licenses)
    _write_external_reports(config, analysis)

    split_records = {name: len(analysis.splits.splits[name]) for name in config.split_names}
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
        checksums_sha256=sha256_file(output / _FULL_CHECKSUMS),
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
            agentic_sources=frozenset(config.splits.agentic_sources),
        ),
    )
    if not splits.report["constraints"]["all_satisfied"]:
        raise ReleaseBuildError("split construction violated one or more leakage constraints")

    enriched: list[dict[str, Any]] = []
    profile_version = "v0.2" if config.schema_version == "0.2" else "v0.1"
    release_schema = (
        config.project_root / "schemas" / f"promptsec-release-record-{profile_version}.schema.json"
    )
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
        "records_by_split": {name: len(splits.splits[name]) for name in config.split_names},
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
    checksum_path = root / _FULL_CHECKSUMS
    if not checksum_path.is_file():
        return [f"missing {_FULL_CHECKSUMS}"]
    failures: list[str] = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"{_FULL_CHECKSUMS}:{line_number}: malformed line")
            continue
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            failures.append(f"{_FULL_CHECKSUMS}:{line_number}: unsafe path {relative}")
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
    *,
    offline: bool,
    source_overrides: Mapping[str, str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    license_entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    configured_source_ids = {SourceConfig.load(path).id for path in config.source_configs}
    unknown_overrides = sorted(set(source_overrides) - configured_source_ids)
    if unknown_overrides:
        raise ReleaseBuildError(f"source overrides have no configured source: {unknown_overrides}")

    for source_path in config.source_configs:
        source = SourceConfig.load(source_path)
        if source.id in seen_sources:
            raise ReleaseBuildError(f"duplicate configured source id: {source.id}")
        if source.id not in config.mapping_quality.profiles:
            raise ReleaseBuildError(f"missing mapping-quality profile for source {source.id}")
        seen_sources.add(source.id)

        artifact_paths = acquire_source(
            source,
            config.paths.raw_dir,
            offline=offline,
            local_path=source_overrides.get(source.id),
        ).artifacts
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
                "acquisition": {
                    "method": source.acquisition.method,
                    "cache_path": source.acquisition.cache_path,
                    "license_file": source.acquisition.license_file,
                    "used_files": list(source.acquisition.used_files),
                    "package_name": source.acquisition.package_name,
                    "package_version": source.acquisition.package_version,
                    "benchmark_version": source.acquisition.benchmark_version,
                    "snapshot_filename": source.acquisition.snapshot_filename,
                    "snapshot_sha256": source.acquisition.snapshot_sha256,
                },
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
    licenses = _license_inventory(config.identity.id, license_entries)
    return records, sources, licenses


def _license_inventory(release_id: str, license_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a component-scoped publication decision without flattening source terms."""

    ordered_entries = sorted(license_entries, key=lambda item: item["manifest"]["source_id"])
    publication_blockers = [
        {
            "source_id": entry["manifest"]["source_id"],
            "scope": component.get("scope"),
            "license_expression": component.get("license_expression"),
            "redistribution": component.get("redistribution"),
            "reason": "Redistribution permission is unresolved in the source manifest.",
        }
        for entry in ordered_entries
        for component in entry["manifest"].get("components", [])
        if component.get("redistribution") == "unknown"
    ]
    payload_status = (
        "BLOCKED_PENDING_LICENSE_REVIEW" if publication_blockers else "MANIFEST_REVIEW_COMPLETE"
    )
    return {
        "schema_version": "0.1",
        "release_id": release_id,
        "notice": (
            "License obligations remain component-specific; NOASSERTION and conditional "
            "entries require release-time review."
        ),
        "publication_status": payload_status,
        "publication_scope": "dataset_payload",
        "publication_components": {
            "repository_code": {
                "status": "PUBLISHABLE",
                "license": "PROJECT_LICENSE",
            },
            "reports_and_metadata": {
                "status": "PUBLISHABLE",
                "contains_source_text": False,
            },
            "dataset_payload": {
                "status": payload_status,
                "contains_source_text": True,
                "distribution": "LOCAL_REBUILD_ONLY"
                if publication_blockers
                else "SOURCE_TERMS_APPLY",
            },
        },
        "publication_blockers": publication_blockers,
        "sources": ordered_entries,
    }


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
        for split_name in config.split_names:
            split_records = [
                analysis.records_by_id[record_id]
                for record_id in analysis.splits.splits[split_name]
            ]
            _write_jsonl(staging / f"{split_name}.jsonl", split_records)

        _write_jsonl(staging / "review_queue.jsonl", analysis.review_queue)
        if config.splits.agentic_sources:
            agentic_queue = [
                entry
                for entry in analysis.review_queue
                if entry.get("source") in config.splits.agentic_sources
            ]
            _write_jsonl(staging / "agentic_review_queue.jsonl", agentic_queue)
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
        _write_json(
            staging / "release_manifest.json",
            _release_manifest(staging, config, analysis),
        )
        _write_named_checksums(
            staging,
            _PIPELINE_CHECKSUMS,
            _PUBLISHABLE_METADATA_FILES,
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
    if config.paths.agentic_review_queue is not None:
        agentic_queue = [
            entry
            for entry in analysis.review_queue
            if entry.get("source") in config.splits.agentic_sources
        ]
        _write_jsonl(config.paths.agentic_review_queue, agentic_queue)


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
    source_revisions = {
        record["metadata"]["dataset_provenance"]["source_dataset"]["id"]: record["metadata"][
            "dataset_provenance"
        ]["source_dataset"]["revision"]
        for record in analysis.records
    }
    lines = [
        f"# {config.identity.title}",
        "",
        f"{config.identity.title} is an **experimental audited release**. It is not a "
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
        "| Source | Records | Pinned revision |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {source} | {count} | {source_revisions.get(source) or 'version pin'} |"
        for source, count in sorted(source_counts.items())
    )
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
    lines.extend(f"| {name} | {split_counts[name]} |" for name in config.split_names)
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
            "mapping rules, or manual review. See the versioned family-mapping document in "
            "`docs/`.",
            "",
            "## Limitations and licensing",
            "",
            "The corpus is source-imbalanced and contains uncertain or non-annotated mappings "
            "explicitly isolated in review queues. Consult `statistics.json` for exact source "
            "shares and `licenses.json` before redistribution; obligations are component-"
            "specific.",
            "",
            "Generated split and review-queue JSONL files are local reconstruction outputs. "
            "They are deliberately ignored by Git and are not redistributed in this release. "
            "The committed release surface contains only redacted statistics, provenance "
            "identifiers, decisions, and checksums.",
            "",
            f"Dataset-payload publication status: **{licenses['publication_status']}**. One or "
            "more source payload components, including the existing BIPIA restriction, remain "
            "NOASSERTION/unknown. Repository code and redacted reports remain separately "
            "classified in `licenses.json`.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/build_dataset.py --config "
            f"{_portable_path(config.path, config.project_root)} --output "
            f"{_portable_path(config.paths.output, config.project_root)} --offline",
            "```",
            "",
            "Local `checksums.sha256` covers every other release file, including ignored "
            "payloads. Tracked `checksums_pipeline.txt` covers only publishable redacted "
            "metadata. With the pinned artifacts and configuration, two builds are byte-"
            "identical. No model is trained by this command.",
            "",
        ]
    )
    if config.splits.agentic_sources:
        lines.extend(
            [
                "",
                "## Agentic provisional evaluation",
                "",
                "InjecAgent and AgentDojo labels are source-derived definitions, not human "
                "gold annotations. Their records are isolated in "
                "`test_agentic_provisional` and `agentic_review_queue.jsonl`; none enter "
                "training. Runtime attack success, model behavior, defenses, and tool "
                "executions are not imported.",
                "",
                "- Agentic sources outside the provisional split: "
                f"{not constraints['no_agentic_source_outside_provisional']}",
                "- Agentic parent-group leakage: "
                f"{not constraints['no_agentic_parent_group_leakage']}",
            ]
        )
    return "\n".join(lines)


def _release_manifest(
    staging: Path,
    config: DatasetReleaseConfig,
    analysis: ReleaseAnalysis,
) -> dict[str, Any]:
    """Describe local payloads without copying any source-derived content."""

    payloads = []
    for path in sorted(staging.glob("*.jsonl"), key=lambda item: item.name):
        records = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        payloads.append(
            {
                "name": path.name,
                "records": records,
                "sha256": sha256_file(path),
                "publication_status": "LOCAL_ONLY_NOT_REDISTRIBUTED",
            }
        )
    return {
        "schema_version": "0.1",
        "release_id": config.identity.id,
        "taxonomy_version": config.identity.taxonomy_version,
        "no_model_training": True,
        "publication_model": "REPRODUCIBLE_METADATA_ONLY",
        "local_payloads": payloads,
        "payload_records_after_exact_deduplication": sum(
            len(record_ids) for record_ids in analysis.splits.splits.values()
        ),
        "review_queue_records": len(analysis.review_queue),
        "publishable_metadata": {
            "status": "PUBLISHABLE",
            "contains_source_text": False,
            "files": [*_PUBLISHABLE_METADATA_FILES, _PIPELINE_CHECKSUMS],
            "checksums": _PIPELINE_CHECKSUMS,
        },
        "local_full_checksums": _FULL_CHECKSUMS,
    }


def _write_named_checksums(root: Path, output_name: str, file_names: tuple[str, ...]) -> None:
    paths = [root / name for name in file_names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ReleaseBuildError(
            f"cannot write {output_name}; missing publishable metadata: {missing}"
        )
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    (root / output_name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != _FULL_CHECKSUMS
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / _FULL_CHECKSUMS).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


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
    extensions = record.get("extensions")
    agentic = extensions.get("agentic_source") if isinstance(extensions, dict) else None
    if source_id == "injecagent" and isinstance(agentic, dict):
        attack_mode = agentic.get("attack_mode")
        attacker_case_id = agentic.get("attacker_case_id")
        if all(isinstance(value, str) and value for value in (attack_mode, attacker_case_id)):
            return f"injecagent:{attack_mode}:{attacker_case_id}"
    if source_id == "agentdojo" and isinstance(agentic, dict):
        suite_id = agentic.get("suite_id")
        injection_task_id = agentic.get("injection_task_id")
        if all(
            isinstance(value, (str, int)) and str(value) for value in (suite_id, injection_task_id)
        ):
            return f"agentdojo:{suite_id}:{injection_task_id}"
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
