# PromptSec-Dataset v0.1 family mapping

This document defines the provenance-backed grouping rules used by the Phase 3.2
release builder. A `template_family` is an experimental grouping key for leakage
control and future out-of-distribution evaluation. It is not a new Taxonomy v1.0
label and does not modify the normative annotation axes.

## Evidence precedence

The builder assigns a family only from structured evidence, in this order:

1. an explicit source label or category preserved in `original_labels`;
2. an upstream generation template or task configuration;
3. a source-specific mapping rule listed below;
4. a manual-review decision, when later available.

Free-text keyword matching is not an accepted source of family identity. Records
without an applicable structured rule receive a documented source-specific fallback
family and remain eligible for review.

For semantic leakage control, two sources also expose a narrower structured group
key. BIPIA records sharing the same `(payload_type, attack_domain)` are kept in one
semantic cluster, and Open-Prompt-Injection variants sharing one `task_config` are
kept in one cluster. This prevents generated variants such as language-translation
directives from being split merely because their surface words differ. PromptInject
and NotInject continue to use lexical-semantic clustering because their collection or
category labels are too broad to assert paraphrase equivalence.

## Source rules

| Source evidence | `template_family` | Rule basis |
|---|---|---|
| PromptInject `collection=goal_hikacking_attacks` | `override_previous_instructions` | Explicit upstream collection; the legacy objective is migrated to `TASK_HIJACKING`. |
| PromptInject `collection=prompt_leaking_attacks` | `prompt_or_policy_disclosure` | Explicit upstream collection; migrated to `PROMPT_OR_POLICY_DISCLOSURE`. |
| Open-Prompt-Injection task configuration | `open_pi_<task>` | Explicit `task` from the allow-listed upstream task configuration. |
| NotInject `category=Technique Queries` | `quoted_attack_hard_negative` | Explicit benign hard-negative category; this does not assert that every instruction axis is resolved. |
| Other NotInject categories | `notinject_<normalized_category>` | Explicit upstream category, retained as a hard-negative family. |
| BIPIA attack-domain/payload pair | Source-specific rule table in `grouping.py` | Explicit `attack_domain` and `payload_type`; no prompt-text inference. |

The BIPIA rules distinguish structured domains such as information retrieval,
content generation, external communication, code, and tool-oriented behavior. They
describe benchmark template provenance; they do not claim human-validated attack
semantics.

## Held-out family

PromptSec-Dataset v0.1 reserves `prompt_or_policy_disclosure` for
`test_held_out_family`. No record with that family may appear in `train`. Semantic
clusters are assigned atomically, so a cluster containing a held-out family cannot
leak a variant into training or another test split.

These splits are experimental because the corpus contains only 627 imported source
records and remains strongly source-imbalanced. NotInject is intentionally weighted
toward evaluation to measure over-defense.
