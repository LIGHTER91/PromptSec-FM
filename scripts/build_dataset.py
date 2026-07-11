#!/usr/bin/env python3
"""Build one source JSONL or a complete audited dataset release."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptsec.data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))
