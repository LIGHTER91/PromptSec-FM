# PromptSec-FM Taxonomy v1.0

**Status:** FROZEN  
**Version:** 1.0  
**Freeze date:** 2026-07-11

This document is the normative source for the PromptSec-FM v1.0 labels. The taxonomy separates linguistic properties, contextual alignment, authority, attack purpose, provenance, annotation uncertainty and operational decisions.

## Design principles

1. A prompt injection is contextual: identical text can be malicious, quoted, pedagogical or legitimate depending on source, addressee, authority and task.
2. Prompt injection and jailbreak are distinct but may co-occur.
3. Ground truth, annotation uncertainty, detector verdict, risk and enforcement are separate objects.
4. Provenance must not be collapsed into a single `trusted/untrusted` field.
5. Character offsets are canonical; BIO tags are generated after tokenization.
6. Benign examples use empty attack arrays rather than a synthetic `SAFE` attack label.

## Frozen core axes

### `instruction_presence`

- `NO_INSTRUCTION`: no directive is formulated or reported.
- `INSTRUCTION_PRESENT`: at least one directive is formulated or reported; this does not imply an attack.
- `UNDETERMINED`: available content is insufficient.

### `instruction_presentation`

- `OPERATIVE`: intended to be followed in the current interaction or execution.
- `QUOTED_OR_REPORTED`: quoted, translated, summarized or described as data.
- `HYPOTHETICAL`: belongs to an example, simulation or conditional scenario.
- `UNKNOWN`: cannot be established.
- `NOT_APPLICABLE`: no instruction is present.

### `instruction_addressee`

- `MODEL_OR_AGENT`
- `HUMAN`
- `EXTERNAL_SYSTEM`
- `UNKNOWN`
- `NOT_APPLICABLE`

### `user_goal_alignment`

- `ALIGNED`: contributes to the legitimate user goal.
- `MISALIGNED`: replaces, diverts, degrades or contradicts the goal.
- `UNDETERMINED`
- `NOT_APPLICABLE`

### `protected_policy_alignment`

- `COMPLIANT`: compatible with protected system/developer policy.
- `CONFLICTING`: attempts to violate, bypass or contradict protected policy.
- `UNDETERMINED`
- `NOT_APPLICABLE`

### `authority_status`

Authority is evaluated for the specific directive and requested behavior.

- `WITHIN_AUTHORITY`: the source may issue this directive within the application-defined scope.
- `OUTSIDE_AUTHORITY`: the directive exceeds that scope.
- `SPOOFED`: the content falsely claims or imitates a more privileged authority.
- `UNKNOWN`
- `NOT_APPLICABLE`

### `attack_families`

Multilabel array; benign examples use `[]`.

- `PROMPT_INJECTION`: unauthorized instruction-boundary crossing.
- `JAILBREAK`: attempted bypass of protected model safety/alignment policy.
- `OTHER_PROMPT_ATTACK`: prompt-mediated attack not represented above.

### `attack_objectives`

Multilabel array:

- `TASK_HIJACKING`
- `POLICY_BYPASS`
- `PROMPT_OR_POLICY_DISCLOSURE`
- `SENSITIVE_DATA_EXFILTRATION`
- `UNAUTHORIZED_ACTION`
- `OUTPUT_OR_DECISION_MANIPULATION`
- `STATE_OR_MEMORY_MANIPULATION`
- `AVAILABILITY_DISRUPTION`
- `SYSTEM_COMPROMISE_OR_MALWARE`
- `FRAUD_OR_IMPERSONATION`
- `OTHER`

## Frozen provenance metadata

These fields describe how content entered the system; they are not attack labels.

### `delivery_mode`

`DIRECT`, `INDIRECT`, `HYBRID`, `UNKNOWN`, `NOT_APPLICABLE`

### `source_role`

`SYSTEM`, `DEVELOPER`, `USER`, `ASSISTANT`, `TOOL`, `EXTERNAL_CONTENT`, `MEMORY`, `UNKNOWN`

### `content_origin`

`CHAT_MESSAGE`, `DOCUMENT`, `WEBPAGE`, `EMAIL`, `TOOL_OUTPUT`, `API_RESPONSE`, `DATABASE_RECORD`, `CODE_REPOSITORY`, `MEMORY_ITEM`, `INTER_AGENT_MESSAGE`, `MEDIA_CONTENT`, `OTHER`, `UNKNOWN`

### `ingestion_path`

`CHAT_INPUT`, `FILE_UPLOAD`, `RETRIEVAL`, `WEB_BROWSING`, `TOOL_CALL`, `MEMORY_RECALL`, `AGENT_MESSAGE`, `OTHER`, `UNKNOWN`

### `modality`

`TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `CODE`, `STRUCTURED_DATA`, `MULTIMODAL`

### `source_integrity`

`VERIFIED`, `UNVERIFIED`, `KNOWN_COMPROMISED`, `UNKNOWN`

`source_integrity` must not directly determine authority or the attack label.

## Frozen span representation

Canonical spans use Unicode character offsets over the exact stored text and the half-open convention `[start, end)`.

Frozen span types:

- `DIRECTIVE`
- `INJECTION_PAYLOAD`
- `AUTHORITY_CLAIM`

Overlapping spans are allowed. BIO/BILOU tags are derived and are never canonical ground truth.

## Frozen annotation-quality labels

### `annotation_status`

- `CONFIRMED`
- `INSUFFICIENT_CONTEXT`
- `DISAGREEMENT`
- `EXCLUDED`

### `annotator_confidence`

Floating-point value in `[0.0, 1.0]`; it is not a model probability.

## Derived prompt-injection verdict

Allowed values:

- `NOT_DETECTED`
- `UNCERTAIN`
- `DETECTED`

Default derivation:

```text
DETECTED when:
  instruction_presence     = INSTRUCTION_PRESENT
  instruction_presentation = OPERATIVE
  instruction_addressee    = MODEL_OR_AGENT
  authority_status         in {OUTSIDE_AUTHORITY, SPOOFED}

UNCERTAIN when DETECTED does not hold and a required axis is
UNDETERMINED/UNKNOWN, or annotation_status is
INSUFFICIENT_CONTEXT/DISAGREEMENT.

NOT_DETECTED otherwise.
```

The verdict is derived and must not replace the core annotations.

## Non-canonical outputs

The following stay outside scientific ground truth:

- risk score and severity;
- generic `SAFE` status;
- `SUSPICIOUS` threshold bands;
- `ALLOW`, `MONITOR`, `SANITIZE`, `ISOLATE`, `HUMAN_REVIEW`, `BLOCK`;
- attack success and runtime consequences.

## Reserved extensions, not frozen in v1.0

- `attack_strategies`
- `evasion_methods`
- `target_assets`
- `impact_dimensions`
- `attack_outcome`
- `risk_level`
- `policy_decision`

## Versioning policy

- v1.x may clarify definitions and add optional metadata without changing frozen identifiers.
- Renaming, removing, splitting or merging a frozen label requires v2.0.
- Every dataset release records `taxonomy_version`.
- Label-changing guideline revisions require a migration note.

## Required annotation pilot

Before publishing dataset v1.0, annotate independently with at least two annotators:

- 100 benign examples;
- 100 direct prompt injections;
- 100 indirect prompt injections;
- 50 jailbreaks;
- 50 hard negatives or context-insufficient examples.

Report agreement per axis, confusion matrices and adjudication reasons.
