#!/usr/bin/env python3
"""Validate PromptSec-FM pilot annotation packs."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ENUMS={
"instruction_presence":{"NO_INSTRUCTION","INSTRUCTION_PRESENT","UNDETERMINED"},
"instruction_presentation":{"OPERATIVE","QUOTED_OR_REPORTED","HYPOTHETICAL","UNKNOWN","NOT_APPLICABLE"},
"instruction_addressee":{"MODEL_OR_AGENT","HUMAN","EXTERNAL_SYSTEM","UNKNOWN","NOT_APPLICABLE"},
"user_goal_alignment":{"ALIGNED","MISALIGNED","UNDETERMINED","NOT_APPLICABLE"},
"protected_policy_alignment":{"COMPLIANT","CONFLICTING","UNDETERMINED","NOT_APPLICABLE"},
"authority_status":{"WITHIN_AUTHORITY","OUTSIDE_AUTHORITY","SPOOFED","UNKNOWN","NOT_APPLICABLE"},
"annotation_status":{"CONFIRMED","INSUFFICIENT_CONTEXT","DISAGREEMENT","EXCLUDED"},
}
FAMILIES={"PROMPT_INJECTION","JAILBREAK","OTHER_PROMPT_ATTACK"}
OBJECTIVES={"TASK_HIJACKING","POLICY_BYPASS","PROMPT_OR_POLICY_DISCLOSURE","SENSITIVE_DATA_EXFILTRATION","UNAUTHORIZED_ACTION","OUTPUT_OR_DECISION_MANIPULATION","STATE_OR_MEMORY_MANIPULATION","AVAILABILITY_DISRUPTION","SYSTEM_COMPROMISE_OR_MALWARE","FRAUD_OR_IMPERSONATION","OTHER"}
SPAN_TYPES={"DIRECTIVE","INJECTION_PAYLOAD","AUTHORITY_CLAIM"}

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--items",required=True); ap.add_argument("--annotations",required=True)
    ap.add_argument("--require-complete",action="store_true")
    args=ap.parse_args()
    items={x["id"]:x for x in rows(args.items)}
    anns=rows(args.annotations); seen=set(); errors=[]
    for row in anns:
        iid=row.get("id")
        if iid in seen: errors.append(f"{iid}: duplicate")
        seen.add(iid)
        if iid not in items: errors.append(f"{iid}: unknown id"); continue
        a=row.get("annotations",{})
        for field,allowed in ENUMS.items():
            value=a.get(field)
            if args.require_complete and value is None: errors.append(f"{iid}: missing {field}")
            elif value is not None and value not in allowed: errors.append(f"{iid}: invalid {field}={value}")
        for field,allowed in [("attack_families",FAMILIES),("attack_objectives",OBJECTIVES)]:
            value=a.get(field)
            if args.require_complete and value is None: errors.append(f"{iid}: missing {field}")
            if value is not None:
                bad=set(value)-allowed
                if bad: errors.append(f"{iid}: invalid {field}: {sorted(bad)}")
                if len(value)!=len(set(value)): errors.append(f"{iid}: duplicate values in {field}")
        conf=a.get("annotator_confidence")
        if args.require_complete and conf is None: errors.append(f"{iid}: missing annotator_confidence")
        elif conf is not None and not (0<=conf<=1): errors.append(f"{iid}: confidence outside [0,1]")
        text=items[iid]["content"]["text"]
        for s in a.get("spans") or []:
            if s.get("type") not in SPAN_TYPES: errors.append(f"{iid}: invalid span type")
            if not isinstance(s.get("start"),int) or not isinstance(s.get("end"),int) or not (0<=s["start"]<s["end"]<=len(text)):
                errors.append(f"{iid}: invalid span offsets {s}")
    missing=set(items)-seen
    if missing: errors.append(f"missing IDs: {len(missing)}")
    if len(items)!=400: errors.append(f"items file contains {len(items)}, expected 400")
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"validation failed with {len(errors)} error(s)")
    print(f"OK: {len(items)} items and {len(anns)} annotations validated")
if __name__=="__main__": main()
