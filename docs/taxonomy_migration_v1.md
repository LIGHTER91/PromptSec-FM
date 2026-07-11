# Migration to PromptSec-FM Taxonomy v1.0

This note maps the initial draft taxonomy to the frozen v1.0 structure. Migration is not always one-to-one because the draft mixed semantics, context, attack purpose and operational decisions.

## Core relation labels

| Draft value | v1.0 migration |
|---|---|
| `NO_INSTRUCTION` | `instruction_presence=NO_INSTRUCTION`; presentation, addressee, alignments and authority become `NOT_APPLICABLE`. |
| `ALIGNED_INSTRUCTION` | Split into presence, presentation, addressee, user-goal alignment, protected-policy alignment and authority. |
| `MISALIGNED_INSTRUCTION` | Usually `user_goal_alignment=MISALIGNED`; do not infer authority or attack family without context. |
| `AMBIGUOUS_INSTRUCTION` | Remove. Use `UNDETERMINED`/`UNKNOWN` on affected axes plus `annotation_status`. |

## Global labels

| Draft value | v1.0 migration |
|---|---|
| `SAFE` | Remove as scientific truth; derived `prompt_injection_verdict=NOT_DETECTED`. |
| `SUSPICIOUS` | Remove as truth; use `UNCERTAIN` only for unresolved context. |
| `INJECTION` | Use `attack_families` and derive the verdict from instruction/authority axes. |

## Channel migration

| Draft value | v1.0 migration |
|---|---|
| `DIRECT` | `delivery_mode=DIRECT` |
| `INDIRECT_DOCUMENT` | `delivery_mode=INDIRECT`, `content_origin=DOCUMENT` |
| `INDIRECT_WEB` | `delivery_mode=INDIRECT`, `content_origin=WEBPAGE` |
| `INDIRECT_EMAIL` | `delivery_mode=INDIRECT`, `content_origin=EMAIL` |
| `TOOL_OUTPUT` | `content_origin=TOOL_OUTPUT`, `ingestion_path=TOOL_CALL` |
| `RETRIEVAL` | `ingestion_path=RETRIEVAL`; origin is separate |
| `MEMORY` | `content_origin=MEMORY_ITEM`, `ingestion_path=MEMORY_RECALL` |
| `MULTIMODAL` | `modality=MULTIMODAL`; delivery and origin remain separate |

## Objective migration

| Draft value | v1.0 destination |
|---|---|
| `GOAL_HIJACKING` | `TASK_HIJACKING` |
| `INSTRUCTION_OVERRIDE` | future `attack_strategies`; often paired with `TASK_HIJACKING` or `POLICY_BYPASS` |
| `AUTHORITY_SPOOFING` | `authority_status=SPOOFED`; optional future strategy |
| `PROMPT_EXTRACTION` | `PROMPT_OR_POLICY_DISCLOSURE` |
| `SECRET_EXFILTRATION` | `SENSITIVE_DATA_EXFILTRATION` |
| `TOOL_ABUSE` | `UNAUTHORIZED_ACTION` |
| `POLICY_BYPASS` | `POLICY_BYPASS` |
| `MEMORY_POISONING` | `STATE_OR_MEMORY_MANIPULATION` |
| `RETRIEVAL_POISONING` | usually state/memory or output/decision manipulation, with provenance metadata |
| `OUTPUT_MANIPULATION` | `OUTPUT_OR_DECISION_MANIPULATION` |
| `DENIAL_OF_SERVICE` | `AVAILABILITY_DISRUPTION` |
| `PERSISTENCE` | remove from objectives; model later as impact scope or runtime outcome |

## Trust, severity and decisions

- Remove `TRUSTED/UNTRUSTED`; use `source_integrity` and `authority_status`.
- Move severity, risk score and enforcement decisions outside ground-truth annotations.
- Preserve legacy targets and techniques under `extensions` until their vocabularies are validated.

## Migration requirements

A migration script must:

1. retain the original dataset and label values;
2. record `taxonomy_version=1.0`;
3. mark non-deterministic mappings for human review;
4. never infer authority from `UNVERIFIED`;
5. never infer an attack solely from a technique keyword;
6. validate each item against `schemas/promptsec-annotation-v1.schema.json`.
