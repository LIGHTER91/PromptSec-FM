# PromptSec-FM Pilot v0.1 — Agreement Report

## Study metadata

- Taxonomy version: 1.0
- Pilot version: 0.1
- Annotation dates:
- Annotator A profile:
- Annotator B profile:
- Independence statement:
- Adjudicator:

## Corpus

- Total: 400
- English: 200
- French: 200
- Benign: 100
- Direct prompt injection: 100
- Indirect prompt injection: 100
- Jailbreak: 50
- Hard negative / insufficient context: 50

## Completion and exclusions

| Measure | A | B |
|---|---:|---:|
| Completed | | |
| Excluded | | |
| Insufficient context | | |

## Agreement by single-label axis

| Axis | Valid N | Observed agreement | Cohen κ | Threshold | Pass |
|---|---:|---:|---:|---:|---|
| instruction_presence | | | | 0.80 | |
| instruction_presentation | | | | 0.80 | |
| instruction_addressee | | | | 0.80 | |
| user_goal_alignment | | | | 0.70 | |
| protected_policy_alignment | | | | 0.70 | |
| authority_status | | | | 0.70 | |
| annotation_status | | | | descriptive | |

## Multilabel agreement

| Axis | Exact agreement | Mean Jaccard | Micro F1 | Threshold | Pass |
|---|---:|---:|---:|---:|---|
| attack_families | | | | Jaccard 0.90 | |
| attack_objectives | | | | Jaccard 0.75 | |

## Span agreement

| Measure | Result | Threshold | Pass |
|---|---:|---:|---|
| Exact span-set agreement | | descriptive | |
| Span presence agreement | | descriptive | |
| Mean best-match IoU | | 0.75 | |
| Injection payload IoU | | 0.75 | |

## Stratified analysis

Report agreement separately for:

- language;
- pilot stratum;
- direct versus indirect delivery;
- content origin;
- hard negatives.

## Main disagreement patterns

1.
2.
3.

## Adjudication outcome

- Number of items requiring adjudication:
- Unresolved after adjudication:
- Guideline clarifications:
- Edge cases added to public documentation:

## Decision

Choose one:

- [ ] Keep taxonomy v1.0 unchanged.
- [ ] Clarify guidelines in v1.x without changing label meaning.
- [ ] Propose a breaking taxonomy v2.0.
