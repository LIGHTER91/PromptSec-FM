from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from promptsec.data.cli import _source_paths, make_parser


def test_fetch_cli_accepts_offline_and_local_path_aliases() -> None:
    parser = make_parser()

    local = parser.parse_args(
        [
            "fetch",
            "--config",
            "source.toml",
            "--offline",
            "--local-path",
            "checkout",
        ]
    )
    alias = parser.parse_args(["fetch", "--config", "source.toml", "--source-path", "checkout"])

    assert local.offline is True
    assert local.local_path == Path("checkout")
    assert alias.local_path == Path("checkout")


def test_release_build_cli_accepts_repeatable_source_paths() -> None:
    args = make_parser().parse_args(
        [
            "build",
            "--config",
            "dataset.yaml",
            "--output",
            "release",
            "--offline",
            "--source-path",
            "injecagent=checkout",
            "--source-path",
            "agentdojo=cache",
        ]
    )

    assert args.offline is True
    assert _source_paths(args.source_path) == {
        "injecagent": Path("checkout"),
        "agentdojo": Path("cache"),
    }


def test_source_path_parser_rejects_duplicates() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate source_id"):
        _source_paths(["injecagent=one", "injecagent=two"])
