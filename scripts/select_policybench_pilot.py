#!/usr/bin/env python3
"""Create a text-free deterministic stratified PolicyBench selection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.data.hashing import sha256_file  # noqa: E402
from promptsec.policybench.blueprints import (  # noqa: E402
    build_blueprint_plan,
    policy_descriptors_from_catalogues,
)
from promptsec.policybench.config import PolicyBenchConfig  # noqa: E402
from promptsec.policybench.generation import _counterfactual_plan  # noqa: E402
from promptsec.policybench.io import write_json  # noqa: E402
from promptsec.policybench.policies import load_policy_catalogs  # noqa: E402
from promptsec.policybench.selection import (  # noqa: E402
    PolicyBenchSelectionError,
    build_selection_manifest,
    select_stratified_blueprints,
)


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
        description=(
            "Select exact per-category records from the unchanged deterministic full plan."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="PolicyBench YAML config.")
    parser.add_argument("--output", type=Path, required=True, help="Selection manifest JSON path.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic selection seed.")
    parser.add_argument(
        "--records-per-category",
        type=_positive_integer,
        required=True,
        help="Exact number selected for each frozen generation category.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = PolicyBenchConfig.load(config_path)
        if args.seed != config.seed:
            config = replace(config, seed=args.seed)
        policy_root = (REPOSITORY_ROOT / config.paths.policies).resolve()
        catalogues = load_policy_catalogs(policy_root)
        policies = policy_descriptors_from_catalogues(catalogues)
        base_plan = build_blueprint_plan(config, policies)
        attached_plan, _ = _counterfactual_plan(base_plan, config, policies)
        selected = select_stratified_blueprints(
            attached_plan,
            seed=args.seed,
            records_per_category=args.records_per_category,
        )
        manifest = build_selection_manifest(
            attached_plan,
            selected,
            seed=args.seed,
            records_per_category=args.records_per_category,
            source_config_sha256=sha256_file(config_path),
        )
        write_json(args.output, manifest)
    except (OSError, TypeError, ValueError, PolicyBenchSelectionError) as error:
        print(f"PolicyBench selection failed: {error}", file=sys.stderr)
        return 1
    summary = {
        key: manifest[key]
        for key in (
            "manifest_type",
            "selector",
            "seed",
            "source_plan_records",
            "selected_records",
            "category_counts",
            "domain_counts",
            "language_counts",
            "counterfactual_type_counts",
            "selected_plan_sha256",
        )
    }
    summary["output"] = args.output.as_posix()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
