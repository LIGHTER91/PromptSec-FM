#!/usr/bin/env python3
"""Fetch immutable artifacts declared by a source configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptsec.data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["fetch", *sys.argv[1:]]))
