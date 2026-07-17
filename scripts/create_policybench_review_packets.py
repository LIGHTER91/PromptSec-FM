#!/usr/bin/env python3
"""Create deterministic, blinded A/B review packets from PolicyBench SILVER data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.policybench.review_packets import create_review_packets  # noqa: E402
from promptsec.policybench.validation import iter_dataset_records  # noqa: E402


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create blinded, independently shuffled PolicyBench annotation packets."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Release directory or canonical JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for annotator packets, hidden manifests, and checksums.",
    )
    parser.add_argument(
        "--records",
        type=_positive_integer,
        required=True,
        help="Requested number of records; atomic groups are never split.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Deterministic shuffle seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = iter_dataset_records(args.dataset)
    result = create_review_packets(
        records,
        args.output,
        record_count=args.records,
        seed=args.seed,
    )
    summary = {
        "schema_version": "0.1",
        "phase_state": result.packet_manifest["phase_state"],
        "gold_claim_permitted": False,
        "automatic_gold_records": result.packet_manifest["automatic_gold_records"],
        "requested_records": args.records,
        "selected_records": len(result.selected_ids),
        "output": result.output.as_posix(),
        "files": [
            "annotator_A.jsonl",
            "annotator_B.jsonl",
            "researcher_manifest.json",
            "packet_manifest.json",
            "selection_report.json",
            "checksums.sha256",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
