#!/usr/bin/env python3
"""Validate a local PromptSec-PolicyBench release or canonical JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.policybench.validation import (  # noqa: E402
    iter_dataset_records,
    validate_record_collection,
    validate_release_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PolicyBench canonical schemas, semantics, and SILVER state."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    records = iter_dataset_records(args.dataset)
    report = validate_record_collection(records)
    integrity = validate_release_directory(args.dataset, records)
    report["release_integrity"] = integrity
    if integrity["validation_status"] == "FAIL":
        report["validation_status"] = "FAIL"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        from promptsec.policybench.io import write_json

        write_json(args.report, report)
    print(encoded, end="")
    return 0 if report["validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
