# Annotation Pilot v0.1 — 400 examples

## Objective

Validate whether PromptSec-FM Taxonomy v1.0 can be applied reproducibly by two independent annotators before any stable dataset release.

The pilot tests the axes that the project identifies as central: instruction presence and presentation, addressee, alignment with the user goal, protected-policy alignment, authority, attack family, attack objective and span localization. This follows the project’s scientific position that prompt injection is contextual rather than a text-only binary property.

## Frozen sample

The pilot contains exactly 400 bilingual controlled examples:

| Stratum | Count |
|---|---:|
| Benign | 100 |
| Direct prompt injection | 100 |
| Indirect prompt injection | 100 |
| Jailbreak | 50 |
| Hard negative or insufficient context | 50 |
| **Total** | **400** |

English and French are balanced 200/200. Examples cover finance, health, email, web, code, travel, HR and legal contexts.

This is an **annotation-instrument validation set**, not a benchmark and not a final training corpus. Its controlled templates make edge cases easy to audit, but agreement on this pilot must not be presented as agreement on naturally occurring attacks.

## Files

- `data/pilot_v0.1/items.jsonl`: immutable candidate items without gold labels.
- `annotations/pilot_v0.1/annotator_a.jsonl`: independent blank annotation pack A.
- `annotations/pilot_v0.1/annotator_b.jsonl`: independent blank annotation pack B.
- `annotations/pilot_v0.1/adjudicated.jsonl`: created after independent annotation.
- `scripts/validate_pilot.py`: validates IDs, allowed values, completeness and spans.
- `scripts/compute_agreement.py`: computes agreement per axis.
- `scripts/adjudicate_template.py`: creates a disagreement file for adjudication.

## Independence rules

1. Annotators read `docs/taxonomy.md` and `docs/annotation_guidelines.md` before starting.
2. They annotate separately and do not view the other pack.
3. They must not use a shared LLM-generated answer as ground truth.
4. Discussion starts only after both packs are frozen.
5. Annotator identity may be pseudonymous, but the study report records whether annotators are humans, models or one of each.
6. If LLM annotators are studied, use separate runs and report model/version/prompt/temperature. Do not call two outputs “two human annotators”.

## Recommended workflow

```bash
python scripts/validate_pilot.py   --items data/pilot_v0.1/items.jsonl   --annotations annotations/pilot_v0.1/annotator_a.jsonl   --require-complete

python scripts/validate_pilot.py   --items data/pilot_v0.1/items.jsonl   --annotations annotations/pilot_v0.1/annotator_b.jsonl   --require-complete

python scripts/compute_agreement.py   --a annotations/pilot_v0.1/annotator_a.jsonl   --b annotations/pilot_v0.1/annotator_b.jsonl   --out reports/pilot_v0.1/agreement.json

python scripts/adjudicate_template.py   --a annotations/pilot_v0.1/annotator_a.jsonl   --b annotations/pilot_v0.1/annotator_b.jsonl   --out annotations/pilot_v0.1/adjudication_queue.jsonl
```

## Agreement measures

For each single-label axis:

- observed agreement;
- Cohen’s kappa;
- confusion matrix;
- number of missing values.

For multilabel axes:

- exact-set agreement;
- mean Jaccard similarity;
- micro precision, recall and F1, treating annotator A as reference only for calculation symmetry reporting;
- per-label positive agreement.

For spans:

- exact-set agreement;
- mean best-match character IoU by span type;
- presence/absence agreement.

Cohen’s kappa is not used for multilabel sets or raw spans.

## Acceptance criteria

The pilot passes only when:

- every pack contains the same 400 unique IDs;
- no schema-invalid or out-of-range span remains;
- all core axes have at least 95% complete independent annotations;
- `instruction_presence`, `instruction_presentation`, and `instruction_addressee`: κ ≥ 0.80;
- `user_goal_alignment`, `protected_policy_alignment`, and `authority_status`: κ ≥ 0.70;
- attack-family exact agreement ≥ 0.90 and mean Jaccard ≥ 0.90;
- attack-objective mean Jaccard ≥ 0.75;
- injection-payload span mean IoU ≥ 0.75;
- no category has more than 15% unresolved disagreement after adjudication.

These are project acceptance thresholds, not universal scientific standards. If an axis fails, revise the annotation guideline—not the labels—unless the error analysis demonstrates that the frozen distinction is unusable.

## Required report

`reports/pilot_v0.1/report.md` must include:

- annotator profile and independence statement;
- completion and exclusion counts;
- agreement table per axis;
- confusion matrices;
- results by stratum and language;
- top disagreement patterns;
- adjudication decisions;
- guideline clarifications;
- recommendation: keep v1.0, clarify v1.x, or propose v2.0.

## Data governance

The current 400 items are controlled synthetic examples and contain no real personal data. Future natural examples must preserve source license, provenance, consent/privacy status and any redistribution restrictions.
