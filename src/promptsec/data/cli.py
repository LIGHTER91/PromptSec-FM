"""Command-line interface for fetching, building, and validating datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from promptsec.data.acquisition import acquire_source
from promptsec.data.config import SourceConfig
from promptsec.data.importers.base import load_importer
from promptsec.data.pipeline import build_dataset
from promptsec.data.release import build_release
from promptsec.data.validation import require_valid_record


def _keyed_paths(values: list[str], *, key_name: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"paths must use {key_name}=PATH")
        key, path = value.split("=", 1)
        if not key or not path:
            raise argparse.ArgumentTypeError(f"paths must use {key_name}=PATH")
        if key in parsed:
            raise argparse.ArgumentTypeError(f"duplicate {key_name.lower()} {key!r}")
        parsed[key] = Path(path)
    return parsed


def _inputs(values: list[str]) -> dict[str, Path]:
    return _keyed_paths(values, key_name="ARTIFACT_ID")


def _source_paths(values: list[str]) -> dict[str, Path]:
    return _keyed_paths(values, key_name="SOURCE_ID")


def _iter_json_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected an object")
                    yield value
        return
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        yield from value
    else:
        raise ValueError(f"{path}: expected an object or array")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="promptsec-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download and checksum configured artifacts")
    fetch.add_argument("--config", required=True, type=Path)
    fetch.add_argument("--destination", default=Path("data/raw"), type=Path)
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument("--offline", action="store_true")
    fetch.add_argument(
        "--local-path",
        "--source-path",
        dest="local_path",
        type=Path,
        help="use a local file or checkout while still enforcing its configured pin",
    )

    build = subparsers.add_parser(
        "build", help="build one source or a complete audited YAML release"
    )
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--input", action="append", default=[], metavar="ARTIFACT_ID=PATH")
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--report", type=Path)
    build.add_argument("--imported-at")
    build.add_argument("--offline", action="store_true")
    build.add_argument(
        "--source-path",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="override one release source with a checksum- or revision-verified local path",
    )

    validate = subparsers.add_parser("validate", help="validate canonical JSON or JSONL")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "fetch":
        config = SourceConfig.load(args.config)
        result = acquire_source(
            config,
            args.destination,
            overwrite=args.overwrite,
            offline=args.offline,
            local_path=args.local_path,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "build":
        if args.config.suffix.lower() in {".yaml", ".yml"}:
            if args.input or args.report or args.imported_at:
                raise ValueError(
                    "release YAML builds do not accept --input, --report, or --imported-at; "
                    "these values are pinned by the release configuration"
                )
            report = build_release(
                args.config,
                output_override=args.output,
                offline=args.offline,
                source_overrides=_source_paths(args.source_path),
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.offline or args.source_path:
            raise ValueError(
                "single-source TOML builds use explicit --input paths; "
                "--offline and --source-path apply to release YAML builds"
            )
        config = SourceConfig.load(args.config)
        importer = load_importer(config.importer, config, imported_at=args.imported_at)
        report = build_dataset(
            importer,
            _inputs(args.input),
            args.output,
            report_path=args.report,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    count = 0
    for record in _iter_json_records(args.path):
        require_valid_record(record)
        count += 1
    print(f"OK: {count} canonical record(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
