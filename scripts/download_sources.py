#!/usr/bin/env python3
"""Acquire pinned PromptSec sources once for subsequent offline release builds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.data.acquisition import acquire_sources  # noqa: E402
from promptsec.data.config import SourceConfig  # noqa: E402
from promptsec.data.release_config import DatasetReleaseConfig  # noqa: E402


def _source_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("local paths must use SOURCE_ID=PATH")
        source_id, path = value.split("=", 1)
        if not source_id or not path:
            raise argparse.ArgumentTypeError("local paths must use SOURCE_ID=PATH")
        if source_id in parsed:
            raise argparse.ArgumentTypeError(f"duplicate source id {source_id!r}")
        parsed[source_id] = Path(path)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify immutable source artifacts. After this command succeeds, "
            "repeat it with --offline to audit cache completeness without network access."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="release YAML (all configured sources) or one source TOML",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="raw cache root; defaults to paths.raw_dir for a release YAML",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="acquire only this configured source id (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="acquire every source declared by the release configuration",
    )
    parser.add_argument(
        "--local-path",
        "--source-path",
        dest="local_paths",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="use a local file or checkout while still enforcing its immutable pin",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        release = DatasetReleaseConfig.load(config_path)
        configs = [SourceConfig.load(path) for path in release.source_configs]
        destination = args.destination or release.paths.raw_dir
    elif config_path.suffix.lower() == ".toml":
        configs = [SourceConfig.load(config_path)]
        destination = args.destination or REPOSITORY_ROOT / "data" / "raw"
    else:
        raise ValueError("--config must be a release YAML or source TOML")

    if args.all and args.source:
        raise ValueError("--all cannot be combined with --source")

    results = acquire_sources(
        configs,
        destination,
        source_ids=None if args.all or not args.source else args.source,
        overwrite=args.overwrite,
        offline=args.offline,
        local_paths=_source_paths(args.local_paths),
    )
    print(
        json.dumps(
            {
                "destination": Path(destination).resolve().as_posix(),
                "network_access": "disabled" if args.offline else "allowed_if_cache_missing",
                "sources": {
                    source_id: result.to_dict() for source_id, result in sorted(results.items())
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
