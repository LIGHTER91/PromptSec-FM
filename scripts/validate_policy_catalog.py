#!/usr/bin/env python3
"""Validate all authored PromptSec-PolicyBench policy catalogues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promptsec.data.hashing import sha256_file  # noqa: E402
from promptsec.policybench.policies import load_policy_catalogs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the six bilingual PromptSec-PolicyBench policy catalogues."
    )
    parser.add_argument("--policies", type=Path, required=True)
    args = parser.parse_args(argv)

    catalogues = load_policy_catalogs(args.policies)
    files = sorted(args.policies.glob("*.yaml"), key=lambda path: path.name)
    result = {
        "schema_version": "0.1",
        "status": "VALID",
        "catalogues": len(catalogues),
        "policies": sum(len(catalogue["policies"]) for catalogue in catalogues.values()),
        "domains": {
            domain: len(catalogue["policies"]) for domain, catalogue in sorted(catalogues.items())
        },
        "checksums": {path.name: sha256_file(path) for path in files},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
