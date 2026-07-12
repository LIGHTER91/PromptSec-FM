#!/usr/bin/env python3
"""Export pinned AgentDojo definitions without running its benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.data.agentdojo_snapshot import (  # noqa: E402
    AgentDojoSnapshotError,
    export_snapshot,
    snapshot_sha256,
)
from promptsec.data.config import SourceConfig  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a canonical AgentDojo task-definition snapshot through get_suites only. "
            "The pinned package must be installed in this interpreter."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "sources" / "agentdojo.toml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SourceConfig.load(args.config)
    acquisition = config.acquisition
    if not acquisition.package_version or not acquisition.benchmark_version:
        raise AgentDojoSnapshotError(
            "AgentDojo config must define acquisition package_version and benchmark_version"
        )
    if not acquisition.snapshot_filename or not config.revision:
        raise AgentDojoSnapshotError(
            "AgentDojo config must define snapshot_filename and source revision"
        )
    repository_root = config.path.parents[2]
    output = args.output or (
        repository_root / acquisition.cache_path / acquisition.snapshot_filename
    )
    snapshot = export_snapshot(
        output,
        acquisition.package_version,
        config.revision,
        acquisition.benchmark_version,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "benchmark_version": acquisition.benchmark_version,
                "counts": snapshot["counts"],
                "no_benchmark_execution": True,
                "output": output.resolve().as_posix(),
                "package_version": acquisition.package_version,
                "revision": config.revision,
                "sha256": snapshot_sha256(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
