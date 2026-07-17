# PromptSec-PolicyBench v0.1 limitations

## Synthetic policy realism

The protected policies are structured research instruments, not copies of confidential production
system or developer messages. They may omit operational exceptions, legal constraints, delegation
chains, or organization-specific language. Performance on them does not prove robustness on real
production prompts.

## Generator artifacts

An LLM or deterministic template can introduce repetitive syntax, category-specific phrasing,
translation artifacts, or provider signatures. Models may learn these shortcuts instead of policy
reasoning. Local lexical-semantic deduplication reduces but cannot eliminate semantic duplication or
shared style.

## Automatic semantic validation

Closed schemas and declared evidence can detect many deviations, but deterministic checks cannot
prove full natural-language equivalence. An advisory validation model shares generator biases and is
not independent gold truth. Human review remains necessary.

## Blueprint assumptions

Blueprint labels are consequences of authored structured rules. Errors or contradictions in those
rules can propagate consistently through many records. Catalogue validation catches duplicate IDs,
unknown actions, simple unconditional contradictions, empty text, missing authority boundaries, and
unsupported values; it cannot prove that every policy is complete, fair, lawful, or realistic.

## Frozen annotation-status tension

The canonical v1.0 schema requires `annotation_status` and `annotator_confidence` but has no value for
"automatically blueprint-derived, pending human review." Context-complete generated records use
`annotation_status=CONFIRMED` only to mean internally unambiguous and `annotator_confidence=0.0`
because no human annotator exists. The authoritative quality state is the PolicyBench extension:
SILVER and PENDING. Consumers that ignore the extension could misinterpret this distinction.

The extension schema reserves the paired `GOLD_HUMAN_CONFIRMED`/`CONFIRMED` lifecycle state for an
externally adjudicated derivative, but no current PolicyBench command imports returned human labels
or mutates generated records into that state. Generated-release validators continue to enforce
SILVER/PENDING for retained records.

## Language coverage

Version 0.1 covers English and French only. Parallel policy wording does not guarantee comparable
pragmatics, politeness, or authority cues. Accents, curly apostrophes, decomposed Unicode, and other
offset cases are tested, but the released distribution cannot represent every regional variety or
script.

## Counterfactual isolation

Changing a natural-language policy or presentation can unavoidably alter surface tokens beyond the
conceptual variable. The invariant validator proves the declared structured projection, not perfect
causal isolation in human interpretation. Pairwise consistency therefore needs qualitative audit.

Capability changes expose a specific taxonomy tension: available tools affect feasibility but do
not by themselves change permission, policy alignment, or source authority. Because frozen v1.0 has
no feasibility axis, `CAPABILITY_CHANGE` pairs are intentionally label-invariant. They are negative
controls, not label-change tests, and must be reported separately; inventing a label would violate
the frozen taxonomy.

## Duplicate detection

Duplicate checks use deterministic local lexical features. They do not require a network model and
are reproducible, but may miss cross-language paraphrases and meaning-equivalent rewrites or may
cluster lexically similar examples with different semantics. Explicit generation families supply an
additional conservative leakage boundary within language. Counterfactual groups and semantic
duplicate clusters are globally atomic, while policy, scenario, attack, and base-generation family
IDs are language-stratified so language OOD remains possible. Their raw cross-split reuse is reported
and prevents a blanket claim that every raw family ID is globally disjoint; the dedicated
policy-family OOD view separately guarantees that its selected raw families are absent from train.

## Cost and reproducibility

Remote providers can change model revisions, moderation behavior, availability, price, and
determinism. Accepted immutable response artifacts permit a reproducible offline rebuild but do not
make acquisition reproducible. Usage-based cost reports are present only when a provider supplies
usage and configured prices are known.

## Security interpretation

The pipeline treats candidates as inert untrusted data, but downstream notebooks, labellers, or
agent frameworks might not. Never feed the corpus to an agent with enabled tools without an
appropriate isolation boundary. The presence of validator acceptance does not make a candidate safe
to execute.

## Scientific claims

Synthetic PolicyBench results must be reported separately from public-source PromptSec-FM results.
The dataset alone cannot validate the PromptSec-FM scientific hypothesis, establish causal policy
reasoning, certify a defense, or demonstrate robustness to confidential production instructions.
