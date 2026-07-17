#!/usr/bin/env python3
"""Build text-free aggregate quality reports for a PolicyBench release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.policybench.deduplication import (  # noqa: E402
    analyze_policybench_duplicates,
)
from promptsec.policybench.io import read_json_object  # noqa: E402
from promptsec.policybench.reporting import (  # noqa: E402
    build_quality_report,
    write_quality_report,
)
from promptsec.policybench.validation import iter_dataset_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze deterministic local duplicates and write text-free PolicyBench "
            "quality reports."
        )
    )
    parser.add_argument("dataset", type=Path, help="Release directory or canonical JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for quality_report.json, quality_report.md, and checksums.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = iter_dataset_records(args.dataset)
    duplicates = analyze_policybench_duplicates(records)
    split_path = args.dataset / "split_report.json" if args.dataset.is_dir() else None
    split_report = (
        read_json_object(split_path) if split_path is not None and split_path.is_file() else None
    )
    report = build_quality_report(records, duplicates, split_report=split_report)
    written = write_quality_report(report, args.output)
    result = {
        "schema_version": "0.1",
        "status": "WRITTEN",
        "records": report["records"],
        "automatic_gold_records": report["automatic_gold_records"],
        **written,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
