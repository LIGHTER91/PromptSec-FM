# PromptSec-PolicyBench annotation protocol

## Purpose

Human review determines whether an AI-generated SILVER record can become human-confirmed gold. The
generator, automatic validator, scenario author, and model are not substitutes for independent
annotation.

Annotators follow the frozen PromptSec-FM v1.0 annotation guidelines. This document adds only the
PolicyBench packet and promotion procedure; it does not change any canonical label.

## Blinded packet contents

Annotator A and annotator B receive the same selected records in independently shuffled order. A
packet shows only:

- a blinded identifier;
- the protected policy and legitimate user goal, when available;
- available capabilities;
- the exact candidate text and language;
- delivery mode, source role, content origin, ingestion path, modality, and source integrity; and
- minimal scenario provenance needed to interpret the application boundary.

Packets hide the canonical ID, expected labels, blueprint category, generator prompt and model,
policy rule derivation, counterfactual expected changes, duplicate/split cluster, automatic
validation decision, rejection history, and source labels.

Every annotation form begins empty. An annotator must label instruction presence, presentation,
addressee, both alignment axes, authority, attack families, supported objectives, exact spans,
annotation status, and annotator confidence. Confidence is the human annotator's assessment, not a
model score.

## Selection

Selection is deterministic from the configured seed. It balances domain, language, category,
instruction presentation, user-goal alignment, policy alignment, authority, source role, and
counterfactual type while avoiding redundant members of a semantic duplicate cluster. The selection
report states achieved coverage and any deficiency. Selection does not imply correctness or gold
status.

## Independent annotation

Annotators work without access to one another's packet or the hidden researcher manifest. They use
Python Unicode code-point offsets with half-open intervals `[start, end)` over the exact displayed
candidate. They must not normalize text. Overlapping `DIRECTIVE`, `INJECTION_PAYLOAD`, and
`AUTHORITY_CLAIM` spans are permitted.

Missing policy, goal, source authority, or truncated content must not be guessed. Annotators use
`UNKNOWN`, `UNDETERMINED`, `NOT_APPLICABLE`, and `INSUFFICIENT_CONTEXT` according to the frozen
guideline.

## Adjudication and promotion

The researcher joins returned annotations through the hidden blind-ID manifest, reports agreement
per axis, and adjudicates every disagreement with a written reason. Gold promotion requires:

1. two completed independent annotations;
2. schema-valid spans and frozen labels;
3. documented adjudication of all differing fields;
4. a final human annotation and confidence;
5. successful canonical and provenance checks; and
6. an externally implemented adjudication action that emits a derivative record with
   `data_quality=GOLD_HUMAN_CONFIRMED` and `human_validation_status=CONFIRMED`.

Generation, validation, reporting, splitting, and packet-creation commands are prohibited from
performing step 6. The schema reserves the paired gold/confirmed lifecycle state, but this repository
does not currently provide a PolicyBench command that imports returned annotations, transitions a
generated record to `IN_REVIEW`, or mutates a release into gold. Generated and rebuilt retained
records remain SILVER/PENDING; packet creation only prepares external human work.

## Research reporting

Report inter-annotator agreement separately for each core axis, each multilabel set, and spans.
Document disagreement reasons and counterfactual consistency. Report capability-change pairs as
label-invariance negative controls because feasibility is not permission and no frozen feasibility
label exists. PolicyBench outcomes must be reported separately from public-source PromptSec-FM data.
