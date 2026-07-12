#!/usr/bin/env python3
"""Audit v0.2 and create deterministic, blinded human-review packets."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.data.gold_review import (  # noqa: E402
    audit_corpus,
    create_blinded_packets,
    iter_release_records,
    load_gold_config,
    review_priorities,
    select_candidates,
)
from promptsec.data.hashing import sha256_file  # noqa: E402


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


def _md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_gold_config(args.config)
    records = iter_release_records(args.release)
    review_root = REPOSITORY_ROOT / "reports" / "quality_review"
    license_root = REPOSITORY_ROOT / "reports" / "license_review"
    audit = audit_corpus(records)
    _json(review_root / "corpus_quality_v0.2.json", audit)
    _md(
        review_root / "corpus_quality_v0.2.md",
        "\n".join(
            [
                "# Corpus quality review v0.2",
                "",
                f"Phase state: **{audit['phase_state']}**",
                f"Records audited: **{audit['records']}**",
                "",
                "This is a deterministic structural audit, not a human annotation.",
                "See `corpus_quality_v0.2.json` for machine-readable issue details.",
                "",
                "## Issue counts",
                "",
                "| Issue | Count |",
                "|---|---:|",
                *[f"| {name} | {count} |" for name, count in sorted(audit["issue_counts"].items())],
            ]
        ),
    )
    issues = audit["issues"]
    _jsonl(review_root / "span_errors.jsonl", issues.get("span_error", []))
    _jsonl(review_root / "label_conflicts.jsonl", issues.get("semantic_cluster_conflict", []))
    _jsonl(review_root / "contextual_conflicts.jsonl", issues.get("contextual_conflict", []))
    priorities = review_priorities(records)
    _jsonl(review_root / "review_priority.jsonl", priorities)
    bands = {}
    for item in priorities:
        bands[item["priority_band"]] = bands.get(item["priority_band"], 0) + 1
    _md(
        review_root / "review_priority_summary.md",
        "\n".join(
            [
                "# Review priority summary",
                "",
                f"Phase state: **{audit['phase_state']}**",
                f"Records scored: **{len(priorities)}**",
                "",
                "| Priority band | Records |",
                "|---|---:|",
                *[f"| {name} | {count} |" for name, count in sorted(bands.items())],
            ]
        ),
    )
    selected, selection_report = select_candidates(records, config)
    _json(review_root / "gold_candidate_selection_v0.1.json", selection_report)
    packet_report = create_blinded_packets(selected, args.output, seed=int(config["seed"]))
    _json(args.output / "selection_manifest.json", selection_report)
    _json(args.output / "packet_manifest.json", packet_report)
    _json(license_root / "license_evidence_v0.1.json", _license_evidence())
    _json(license_root / "redistribution_matrix_v0.1.json", _redistribution_matrix())
    print(
        json.dumps(
            {
                "phase_state": "READY_FOR_HUMAN_REVIEW",
                "audited": len(records),
                "selected": len(selected),
                "output": args.output.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _license_evidence() -> dict[str, object]:
    manifests = []
    for path in sorted((REPOSITORY_ROOT / "manifests" / "sources").glob("*.json")):
        if path.name == "source-lock.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        components = []
        for component in value.get("components", []):
            component = dict(component)
            redistribution = component.get("redistribution")
            component["final_publication_status"] = {
                "allowed": "REDISTRIBUTABLE_WITH_ATTRIBUTION",
                "conditional": "REDISTRIBUTABLE_SHARE_ALIKE",
                "unknown": "BLOCKED_PENDING_REVIEW",
                "not-redistributed": "LOCAL_REBUILD_ONLY",
            }.get(redistribution, "NOASSERTION")
            component["review_date"] = "2026-07-12"
            component["reviewer_note"] = "Technical evidence review; not legal advice."
            components.append(component)
        manifests.append(
            {
                "source_id": value.get("source_id"),
                "upstream": value.get("upstream"),
                "license_status": value.get("license_status"),
                "components": components,
                "evidence_file": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "evidence_sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "technical_evidence_audit_not_legal_advice": True,
        "sources": manifests,
    }


def _redistribution_matrix() -> dict[str, object]:
    evidence = _license_evidence()
    rows = []
    for source in evidence["sources"]:
        for component in source.get("components", []):
            redistribution = component.get("redistribution")
            status = {
                "allowed": "REDISTRIBUTABLE_WITH_ATTRIBUTION",
                "conditional": "REDISTRIBUTABLE_SHARE_ALIKE",
                "unknown": "BLOCKED_PENDING_REVIEW",
                "not-redistributed": "LOCAL_REBUILD_ONLY",
            }.get(redistribution, "NOASSERTION")
            rows.append(
                {
                    "source_id": source.get("source_id"),
                    "scope": component.get("scope"),
                    "license_expression": component.get("license_expression"),
                    "raw_payload": status,
                    "transformed_records": status,
                    "metadata_only": "REDISTRIBUTABLE",
                    "local_rebuild": "allowed",
                    "final_publication_status": status,
                }
            )
    return {
        "schema_version": "0.1",
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "technical_evidence_audit_not_legal_advice": True,
        "components": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
