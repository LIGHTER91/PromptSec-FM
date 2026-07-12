#!/usr/bin/env python3
"""Check human annotation files without manufacturing gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_FIELDS = (
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "attack_families",
    "attack_objectives",
    "spans",
    "annotation_status",
    "annotator_confidence",
)


def _read(path: Path) -> dict[str, dict]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            values[str(value["blind_id"])] = value
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.packet_dir
    paths = {name: root / f"annotations_{name}.jsonl" for name in ("A", "B")}
    result = {
        "schema_version": "0.1",
        "human_annotation_files_present": all(path.is_file() for path in paths.values()),
        "gold_claim_permitted": False,
        "phase_state": "READY_FOR_HUMAN_REVIEW",
        "disagreements": [],
        "missing_fields": [],
    }
    if not result["human_annotation_files_present"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    first, second = (_read(paths["A"]), _read(paths["B"]))
    if set(first) != set(second):
        result["phase_state"] = "BLOCKED_BY_ANNOTATION_QUALITY"
        result["disagreements"] = [{"reason": "annotator_id_sets_differ"}]
    else:
        for blind_id in sorted(first):
            left = first[blind_id].get("annotation", {})
            right = second[blind_id].get("annotation", {})
            missing = [field for field in _FIELDS if field not in left or field not in right]
            if missing:
                result["missing_fields"].append({"blind_id": blind_id, "fields": missing})
            if left != right:
                result["disagreements"].append({"blind_id": blind_id})
        result["phase_state"] = (
            "BLOCKED_BY_ANNOTATION_QUALITY"
            if result["missing_fields"]
            else "READY_FOR_ADJUDICATION"
            if result["disagreements"]
            else "GOLD_LOCAL_ONLY"
        )
        result["gold_claim_permitted"] = result["phase_state"] == "GOLD_LOCAL_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
