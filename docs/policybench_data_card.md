# PromptSec-PolicyBench v0.1 data card

## Dataset summary

PromptSec-PolicyBench v0.1 is an AI-generated, policy-aware SILVER corpus for PromptSec-FM. The
default local build contains 6,000 validated records: 1,000 for each of six domains and an intended
50/50 English/French balance. The full generated release is not committed to Git.

The dataset is designed for studying contextual prompt-injection classification, protected-policy
alignment, scoped authority, exact spans, attack objectives, and counterfactual sensitivity. It is
not human gold truth and is not evidence that a deployed system is secure.

## Source and authorship

Protected policies are synthetic and authored as bilingual structured catalogues. Each domain has at
least twenty definitions with rule, action, confirmation, authority, sensitive-asset, family, and
version provenance. Natural-language scenarios may be realized by an LLM or by the deterministic
mock template provider. The mock is a local, network-free acquisition provider; `--offline` is
reserved for replay from already accepted artifacts. Every realization is checked against a
pre-labelled blueprint.

No public user record or confidential production system prompt is included by the generator. Raw
provider responses remain local and ignored by Git.

## Composition

Domains are banking, email, calendar, file management, web and purchases, and persistent memory.
Languages are English (`en`) and French (`fr`). The default category targets are 12% no instruction,
13% aligned/compliant, 8% aligned/policy-conflicting, 22% misaligned/policy-conflicting, 8%
misaligned/not policy-conflicting, 10% quoted/reported, 7% hypothetical, 8% spoofed authority, 6%
insufficient context, and 6% hard negatives.

These category percentages are planning targets. Counterfactual attachment replaces donors while
preserving the exact 6,000-record total, domain/language quotas, and mandatory
policy/category/language coverage. The deterministic full-plan guard permits at most a 1%-of-corpus
shift in any category, and the achieved counts are reported rather than represented as exact quotas.

Counterfactual coverage includes policy, user-goal, source-role, delegation, capability, and
presentation changes. Capability pairs may be label-invariant: feasibility is not permission or
authority, and the frozen taxonomy has no capability-feasibility annotation. The pipeline records
an empty expected-change list for that negative control rather than inventing a label; capability
results must be reported separately from label-changing pairs.

## Frozen-value coverage and deliberate exclusions

The deterministic full 6,000-blueprint plan covers every frozen value for instruction presence,
instruction presentation, instruction addressee, user-goal alignment, protected-policy alignment,
and authority status. Candidate annotations use only frozen PromptSec-FM v1.0 values. The following
frozen values are deliberately absent from the plan; absence is a scoped scientific choice, not a
schema change.

| Axis | Values present in the full plan | Deliberately absent | Scientific rationale |
|---|---|---|---|
| Attack family | `PROMPT_INJECTION` | `JAILBREAK`, `OTHER_PROMPT_ATTACK` | v0.1 studies direct and indirect prompt injection against an explicit protected policy. It does not synthesize standalone model-safety jailbreaks, and no residual prompt-attack family is needed for these authored scenarios. |
| Attack objective | `TASK_HIJACKING`, `POLICY_BYPASS`, `PROMPT_OR_POLICY_DISCLOSURE`, `SENSITIVE_DATA_EXFILTRATION`, `UNAUTHORIZED_ACTION`, `OUTPUT_OR_DECISION_MANIPULATION`, `STATE_OR_MEMORY_MANIPULATION`, `AVAILABILITY_DISRUPTION`, `SYSTEM_COMPROMISE_OR_MALWARE`, `FRAUD_OR_IMPERSONATION` | `OTHER` | Each attacking blueprint uses directly supported specific objectives; non-attacks use an empty set. Adding `OTHER` would discard available specificity. |
| Annotation status | `CONFIRMED`, `INSUFFICIENT_CONTEXT` | `DISAGREEMENT`, `EXCLUDED` | The absent statuses require a human disagreement or exclusion decision and are not synthesized by generation. A duplicate may have PolicyBench `data_quality=EXCLUDED`, which does not manufacture the frozen annotation-status value. |

Provenance is intentionally restricted to the six authored, text-only routes. The values below
remain legal under the frozen schema for future datasets but are outside v0.1's scenario design.

| Provenance axis | Values present in the full plan | Deliberately absent | Scientific rationale |
|---|---|---|---|
| Delivery mode | `DIRECT`, `INDIRECT` | `HYBRID`, `UNKNOWN`, `NOT_APPLICABLE` | Each synthetic route has one declared direct or indirect delivery boundary; hybrid and unresolved delivery are not simulated. |
| Source role | `DEVELOPER`, `TOOL`, `EXTERNAL_CONTENT`, `MEMORY`, `UNKNOWN` | `SYSTEM`, `USER`, `ASSISTANT` | The protected policy is separate context; candidates are controlled developer requests or untrusted tool, external-content, and memory inputs. An unresolved source boundary supplies `UNKNOWN`; system, ordinary-user, and prior-assistant candidate routes are not modeled. |
| Content origin | `CHAT_MESSAGE`, `DOCUMENT`, `WEBPAGE`, `EMAIL`, `TOOL_OUTPUT`, `MEMORY_ITEM` | `API_RESPONSE`, `DATABASE_RECORD`, `CODE_REPOSITORY`, `INTER_AGENT_MESSAGE`, `MEDIA_CONTENT`, `OTHER`, `UNKNOWN` | The six domain fixtures use only the named chat, document, web, email, tool, and memory origins; no API, database, repository, agent-message, media, residual, or origin-unknown fixture is authored. |
| Ingestion path | `CHAT_INPUT`, `FILE_UPLOAD`, `RETRIEVAL`, `WEB_BROWSING`, `TOOL_CALL`, `MEMORY_RECALL` | `AGENT_MESSAGE`, `OTHER`, `UNKNOWN` | The supported routes map to explicit ingestion mechanisms; inter-agent and residual or unresolved ingestion are not modeled. |
| Modality | `TEXT` | `IMAGE`, `AUDIO`, `VIDEO`, `CODE`, `STRUCTURED_DATA`, `MULTIMODAL` | v0.1 generates strings and code-point spans only. Text that describes code or data is still `TEXT`; no non-text payload is materialized. |
| Source integrity | `VERIFIED`, `UNVERIFIED`, `KNOWN_COMPROMISED` | `UNKNOWN` | Every authored route assigns an integrity state. Missing source-role context remains `UNVERIFIED` rather than asserting that integrity itself is unknown. |

## Quality and labels

The LLM does not choose labels. A structured scenario blueprint fixes deterministic expected
annotations before generation. Validators check response schema, semantic evidence, language,
authority claims, action preservation, label leakage, exact spans, counterfactual invariants,
duplicates, canonical schema conformance, and verdict derivation.

Automatic validation can establish consistency with an authored synthetic scenario, not empirical
truth. Records remain `SILVER_TEMPLATE`, `SILVER_GENERATED`, or `SILVER_VALIDATED` with
`human_validation_status=PENDING`. Only human double annotation and adjudication can produce
`GOLD_HUMAN_CONFIRMED`.

The schema reserves `GOLD_HUMAN_CONFIRMED` only together with
`human_validation_status=CONFIRMED` for an externally adjudicated derivative. No current
PolicyBench command imports returned annotations or mutates the generated release into that state;
release validation rejects automatic gold or any retained generated record that is not
SILVER/PENDING. Automatically rejected duplicate records can instead be marked
`data_quality=EXCLUDED` while remaining human-validation `PENDING`.

Quality reports include achieved coverage, length percentiles, acceptance, rejection, and retry
rates, rejection reasons, failed validation checks, and span-specific failure and rejection counts
and rates. Optional usage and cost totals are reported only when provider metadata supports them.

## Splits

The release provides train, validation, policy-family OOD, domain OOD, language OOD,
counterfactual-test, and human-review-candidate files. Assignment is deterministic and closes
transitive linkage before allocating components. Counterfactual groups and semantic duplicate
clusters are globally atomic, so those siblings cannot cross any split.

Policy, scenario-template, attack-template, and base-generation family identifiers are namespaced by
language for linkage only. This deliberate exception prevents every bilingual family from joining
English and French into one component and makes a genuine language-OOD view possible. Raw family
identifiers can therefore occur in more than one split; `raw_leakage_values_by_field` reports that
reuse separately and it must not be described as absent. The policy-family OOD guard is stricter for
its evaluation target: every raw policy family selected for `test_policy_family_ood` is excluded
from train even across languages.

The splitter rejects a full build when transitive closure collapses too much of the corpus, a
requested split is empty, a requested split lacks enough independent components, or an allocated
record target deviates by more than one largest component. Held-out domain and language records are
also required to be absent from train. These checks constrain split construction but do not turn raw
bilingual family reuse in non-train views into global atomicity.

## Intended uses

- train or evaluate policy-aware PromptSec-FM components;
- measure separation of goal alignment and policy alignment;
- test authority and presentation counterfactuals;
- evaluate Unicode character-span extraction; and
- select diverse candidates for human annotation.

## Out-of-scope uses

- treating generated labels as human ground truth;
- claiming safety certification or production robustness;
- executing candidate instructions, code, URLs, or tool calls;
- using the corpus as a source of real financial, email, calendar, file, purchase, or memory policy;
  or
- merging PolicyBench and public-source metrics without separate reporting.

## Sensitive and malicious content

Candidates intentionally describe unauthorized money movement, exfiltration, deletion, policy
disclosure, impersonation, memory manipulation, and similar attacks. These are inert research-data
strings. The pipeline never executes them. Users should retain this boundary in downstream tooling.

## Citation and release status

This repository is an active research project. Record the release ID, configuration hash, policy
catalogue hashes, accepted-artifact manifest, split manifest, and code revision in experimental
reports. See the limitation and reproducibility documents before publication.
