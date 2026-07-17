# PromptSec-PolicyBench v0.1 design

## Status and scope

PromptSec-PolicyBench v0.1 is a reproducible synthetic extension of PromptSec-FM. It adds an
explicit protected system/developer policy to contextual prompt-security records. The release is
AI-generated **SILVER data**, not human gold truth. Every generated annotation remains pending
human validation even when deterministic checks confirm that it matches its scenario blueprint.

The design leaves PromptSec-FM Taxonomy v1.0 unchanged. Frozen labels, character-span types, and
the prompt-injection verdict are imported from the existing canonical implementation.

## Scientific objective

Public prompt-injection corpora commonly contain a goal, payload, source, or agent capability but
not the policy whose boundary is allegedly crossed. PolicyBench makes that policy explicit so the
same candidate can be studied along separate axes:

- instruction presence, presentation, and addressee;
- alignment with the legitimate user goal;
- alignment with a protected policy;
- scoped source authority;
- attack family and directly supported objectives;
- exact, possibly overlapping character spans;
- contextual uncertainty; and
- counterfactual label changes caused by one declared context change, plus the explicitly documented
  `CAPABILITY_CHANGE` label-invariance negative control.

Synthetic performance must be reported separately from performance on public-source PromptSec-FM
records. It does not establish robustness to confidential production system prompts.

## Layered architecture

The pipeline deliberately separates objects that an LLM could otherwise conflate:

1. Authored policy catalogues define bilingual policy wording and structured rules.
2. Deterministic scenario blueprints choose a policy, context, category, action, and expected frozen
   annotations before linguistic realization.
3. A provider realizes only natural-language fields and exact substring anchors.
4. Structural and semantic validators compare the untrusted response with the blueprint.
5. The record builder computes spans over the exact final text, derives the canonical verdict, and
   constructs canonical dataset provenance.
6. Deduplication and transitive grouping create leakage-resistant splits.
7. Quality reporting describes coverage, acceptance/rejection/retry rates, span rejections, and
   optional usage cost.
8. Blinded packets start a separate human double-annotation and adjudication workflow.

The provider cannot freely choose labels. An advisory model-based semantic check, when configured,
never replaces deterministic checks or human review.

## Canonical record profile

`schemas/promptsec-policybench-record-v0.1.schema.json` composes the existing dataset-record v0.1
profile. Canonical context, content, annotations, derived verdict, and dataset provenance therefore
remain compatible with existing validation and annotation tooling.

The canonical `metadata` object is closed by the existing dataset-record profile. PolicyBench data
is consequently stored under `extensions.policybench_v0_1`, including:

- `data_quality` and `human_validation_status`;
- policy, scenario, template-family, and counterfactual provenance;
- generator and immutable-artifact metadata;
- validation evidence and quality grouping; and
- split assignment.

The extension is non-canonical metadata. It does not add or rename a taxonomy label.

## Quality-state invariant

Automatically built records may use only `SILVER_TEMPLATE`, `SILVER_GENERATED`,
`SILVER_VALIDATED`, or `EXCLUDED`. They always begin with `human_validation_status=PENDING`.
The schema reserves `data_quality=GOLD_HUMAN_CONFIRMED` only together with
`human_validation_status=CONFIRMED` so an externally adjudicated derivative can be represented.
Generation, rebuild, release validation, reporting, splitting, and packet commands do not emit that
pair. No current PolicyBench command imports returned annotations or mutates an existing generated
release into gold.

For context-complete blueprints, frozen `annotation_status=CONFIRMED` means that the annotation is
unambiguous under the authored scenario. It is not a claim of human confirmation. Because there is
no human annotator at generation time, `annotator_confidence` is fixed at `0.0`; model confidence or
provider probabilities are never copied into that field. Insufficient-context blueprints use the
frozen uncertainty values and `annotation_status=INSUFFICIENT_CONTEXT`.

## Split-linkage contract

Counterfactual groups and semantic duplicate clusters are globally atomic. Policy,
scenario-template, attack-template, and base-generation family linkage is instead namespaced by
language so English and French can form a meaningful language-OOD partition. Raw family IDs may
therefore appear in multiple splits and are disclosed separately in the split report; only selected
raw policy families in the dedicated policy-family OOD view are guaranteed absent from train.

The splitter guards against an oversized transitive component, empty or underpowered requested
splits, and target deviations larger than one component. These qualifications are part of the
leakage contract and prevent describing all raw family identifiers as globally disjoint.

## Domains and extensibility

Version 0.1 covers banking, email, calendar, file management, web and purchases, and persistent
memory. Each version-controlled catalogue has at least twenty policies with English and French
wording, structured rules, actions, confirmation requirements, authority boundaries, sensitive
assets, families, and versions.

Language is a scenario dimension and a canonical `content.language` value. Adding a language
requires catalogue or realization support and configuration changes, not a canonical record-schema
change.

## Security boundary

Generated text is untrusted data. The pipeline uses strict JSON schemas and UTF-8 decoding, limits
response and record sizes, confines computed paths, and never evaluates generated code or executes
requested tools. Generated text is passed as data, not interpolated into shell commands. Provider
prompts receive a selected blueprint and public policy material only; environment variables,
repository secrets, and API keys are never included.
