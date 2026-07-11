#!/usr/bin/env python3
"""Compute inter-annotator agreement for PromptSec-FM pilot v0.1."""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path

SINGLE = [
    "instruction_presence", "instruction_presentation", "instruction_addressee",
    "user_goal_alignment", "protected_policy_alignment", "authority_status",
    "annotation_status",
]
MULTI = ["attack_families", "attack_objectives"]

def load(path):
    out={}
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        if row["id"] in out: raise ValueError(f"duplicate id {row['id']} in {path}:{n}")
        out[row["id"]]=row["annotations"]
    return out

def kappa(xs,ys):
    pairs=[(x,y) for x,y in zip(xs,ys) if x is not None and y is not None]
    if not pairs: return None
    po=sum(x==y for x,y in pairs)/len(pairs)
    cx=collections.Counter(x for x,_ in pairs); cy=collections.Counter(y for _,y in pairs)
    labels=set(cx)|set(cy)
    pe=sum(cx[l]/len(pairs)*cy[l]/len(pairs) for l in labels)
    return 1.0 if pe==1.0 and po==1.0 else ((po-pe)/(1-pe) if pe<1 else None)

def single_stats(a,b,field,ids):
    xs=[a[i].get(field) for i in ids]; ys=[b[i].get(field) for i in ids]
    valid=[(x,y) for x,y in zip(xs,ys) if x is not None and y is not None]
    labels=sorted(set(x for x,y in valid)|set(y for x,y in valid))
    matrix={x:{y:0 for y in labels} for x in labels}
    for x,y in valid: matrix[x][y]+=1
    return {
        "n_valid":len(valid),"n_missing":len(ids)-len(valid),
        "observed_agreement":sum(x==y for x,y in valid)/len(valid) if valid else None,
        "cohens_kappa":kappa(xs,ys),"confusion_matrix":matrix,
    }

def multi_stats(a,b,field,ids):
    pairs=[]
    tp=fp=fn=0
    label_counts=collections.defaultdict(lambda:[0,0,0,0]) # both,a only,b only,neither
    universe=set()
    for i in ids:
        sa=set(a[i].get(field) or []); sb=set(b[i].get(field) or [])
        universe |= sa|sb
        pairs.append((sa,sb))
        tp += len(sa&sb); fp += len(sb-sa); fn += len(sa-sb)
    for sa,sb in pairs:
        for l in universe:
            if l in sa and l in sb: label_counts[l][0]+=1
            elif l in sa: label_counts[l][1]+=1
            elif l in sb: label_counts[l][2]+=1
            else: label_counts[l][3]+=1
    p=tp/(tp+fp) if tp+fp else 1.0
    r=tp/(tp+fn) if tp+fn else 1.0
    return {
        "n":len(ids),
        "exact_set_agreement":sum(sa==sb for sa,sb in pairs)/len(pairs),
        "mean_jaccard":sum((len(sa&sb)/len(sa|sb) if sa|sb else 1.0) for sa,sb in pairs)/len(pairs),
        "micro_precision":p,"micro_recall":r,
        "micro_f1":2*p*r/(p+r) if p+r else 0.0,
        "per_label_counts":dict(sorted(label_counts.items())),
    }

def iou(x,y):
    inter=max(0,min(x["end"],y["end"])-max(x["start"],y["start"]))
    union=max(x["end"],y["end"])-min(x["start"],y["start"])
    return inter/union if union else 0.0

def span_stats(a,b,ids):
    exact=0; pres=0; values=[]; by_type=collections.defaultdict(list)
    for i in ids:
        aa=a[i].get("spans") or []; bb=b[i].get("spans") or []
        norm=lambda z: sorted((s["start"],s["end"],s["type"]) for s in z)
        exact += norm(aa)==norm(bb)
        pres += bool(aa)==bool(bb)
        for s in aa:
            candidates=[t for t in bb if t["type"]==s["type"]]
            score=max((iou(s,t) for t in candidates),default=0.0)
            values.append(score); by_type[s["type"]].append(score)
        for t in bb:
            if not any(s["type"]==t["type"] for s in aa):
                values.append(0.0); by_type[t["type"]].append(0.0)
    return {
        "exact_set_agreement":exact/len(ids),
        "presence_agreement":pres/len(ids),
        "mean_best_match_character_iou":sum(values)/len(values) if values else 1.0,
        "mean_iou_by_type":{k:sum(v)/len(v) for k,v in sorted(by_type.items())}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True); ap.add_argument("--b",required=True)
    ap.add_argument("--out"); args=ap.parse_args()
    a=load(args.a); b=load(args.b)
    if set(a)!=set(b):
        raise SystemExit(f"ID mismatch: only A={sorted(set(a)-set(b))[:5]}, only B={sorted(set(b)-set(a))[:5]}")
    ids=sorted(a)
    result={
        "n_items":len(ids),
        "single_label":{f:single_stats(a,b,f,ids) for f in SINGLE},
        "multilabel":{f:multi_stats(a,b,f,ids) for f in MULTI},
        "spans":span_stats(a,b,ids),
    }
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out:
        p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text+"\n",encoding="utf-8")
    print(text)
if __name__=="__main__": main()
