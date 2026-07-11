#!/usr/bin/env python3
"""Create a PromptSec-FM adjudication queue from two annotation packs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
FIELDS=["instruction_presence","instruction_presentation","instruction_addressee","user_goal_alignment","protected_policy_alignment","authority_status","attack_families","attack_objectives","spans","annotation_status"]
def load(p):
    return {r["id"]:r["annotations"] for r in (json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip())}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--a",required=True); ap.add_argument("--b",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    a,b=load(args.a),load(args.b)
    if set(a)!=set(b): raise SystemExit("annotation IDs differ")
    q=[]
    for iid in sorted(a):
        diffs={f:{"annotator_a":a[iid].get(f),"annotator_b":b[iid].get(f),"adjudicated":None,"reason":""} for f in FIELDS if a[iid].get(f)!=b[iid].get(f)}
        if diffs:q.append({"id":iid,"disagreements":diffs,"guideline_change_needed":False,"edge_case_candidate":False})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in q)+("\n" if q else ""),encoding="utf-8")
    print(f"wrote {len(q)} disagreement item(s) to {out}")
if __name__=="__main__":main()
