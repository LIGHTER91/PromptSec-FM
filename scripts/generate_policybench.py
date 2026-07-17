#!/usr/bin/env python3
"""Generate or integrity-check a PromptSec-PolicyBench v0.1 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.policybench.config import DOMAIN_ORDER, LANGUAGE_ORDER  # noqa: E402
from promptsec.policybench.generation import (  # noqa: E402
    GenerationOptions,
    PolicyBenchGenerationError,
    generate_policybench,
)


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _non_negative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def _temperature(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 <= result <= 2.0:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible PolicyBench SILVER records; generated data always remains "
            "pending human validation."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="PolicyBench YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Release directory; defaults to the safe output configured in the YAML file.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid every provider call and rebuild only from validated local artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the generation plan without writing artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse integrity-checked validated records and accepted response artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit regeneration instead of reusing an existing release.",
    )
    parser.add_argument(
        "--max-records",
        type=_positive_integer,
        help="Limit generation for a deterministic smoke build.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "Use an integrity-checked deterministic scenario selection; --max-records, "
            "when supplied, must match its exact record count."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAIN_ORDER,
        help="Generate only the selected domains.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGE_ORDER,
        help="Generate only the selected languages.",
    )
    parser.add_argument("--provider", help="Override the configured provider.")
    parser.add_argument("--model", help="Override the configured model identifier.")
    parser.add_argument("--seed", type=int, help="Override the deterministic generation seed.")
    parser.add_argument(
        "--max-retries",
        type=_non_negative_integer,
        help="Override the bounded number of retries after a rejected response.",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_integer,
        help="Override bounded generation concurrency.",
    )
    parser.add_argument(
        "--temperature",
        type=_temperature,
        help="Override generation temperature (0 through 2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")

    options = GenerationOptions(
        offline=args.offline,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
        max_records=args.max_records,
        domains=tuple(args.domains) if args.domains is not None else None,
        languages=tuple(args.languages) if args.languages is not None else None,
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        temperature=args.temperature,
        selection_manifest=args.selection_manifest,
    )
    try:
        result = generate_policybench(args.config, output=args.output, options=options)
    except PolicyBenchGenerationError as error:
        print(f"PolicyBench generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
