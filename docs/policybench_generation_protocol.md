# PromptSec-PolicyBench generation protocol

## Non-gold generation contract

The generator produces AI-generated SILVER candidates. It is not an annotator and is never treated
as a source of human gold truth. Labels come from a validated structured blueprint, not from the
generation model. Human double annotation and adjudication are required for gold promotion.

The record schema reserves `data_quality=GOLD_HUMAN_CONFIRMED` together with
`human_validation_status=CONFIRMED` so an externally adjudicated derivative can be represented.
That schema capability does not authorize automatic promotion: generation and release validation
emit retained records only as SILVER/PENDING, and no PolicyBench command mutates a generated release
into gold.

## Inputs

A generation run is determined by:

- `configs/policybench_v0.1.yaml` (or the loopback-only local example) and command-line overrides;
- the six policy catalogues and their hashes;
- versioned prompt files;
- the global seed and fixed acquisition timestamp;
- provider/model identity and generation parameters; and
- any immutable accepted response artifacts used for an offline rebuild.

No API key is stored in configuration or written to an artifact. A configured environment-variable
name identifies the credential at request time. The local profile allows that variable to be absent
only for a parsed loopback endpoint; remote endpoints still require credentials.

## Blueprint planning

Blueprints are generated deterministically in stable domain, language, category, and index order.
Integer quotas use deterministic largest-remainder apportionment. A blueprint records the selected
policy and rule, user-goal intent, directive action, source provenance, capabilities, category,
expected annotations, span requirements, template families, and optional counterfactual metadata.

All ten v0.1 categories are planned explicitly: no instruction; aligned/compliant; aligned but
policy-conflicting; misaligned and policy-conflicting; misaligned but not policy-conflicting; quoted
or reported; hypothetical; spoofed authority; insufficient context; and hard negatives.

Counterfactual siblings replace deterministic donors from the same domain and language so the plan
remains exactly 6,000 records and preserves required policy/category/language coverage. Same-category
donors are preferred. When coverage forces another donor category, final category counts are
approximate; the full-plan guard rejects a maximum per-category shift greater than 1% of the corpus,
and the achieved distribution is recorded in the manifest and quality report.

## Provider interface

`GenerationProvider` accepts a typed request and returns a typed response with the exact raw response
text, provider identity, model metadata, optional usage, and the fixed generation timestamp.
Implementations are:

- a deterministic mock provider used for local, network-free acquisition in tests and examples; and
- an OpenAI-compatible HTTP adapter whose base URL may point to a remote service or an externally
  managed local model server; and
- a batched `codex_cli` adapter that reuses a normal ChatGPT-authenticated Codex CLI session without
  an API key, isolates each invocation in a temporary directory, and disables local and web tools.

The HTTP adapter uses a timeout, maximum response size, strict UTF-8, closed response shape, and an
explicit model name. Strict JSON Schema, JSON-object, and prompt-constrained JSON modes all feed the
same closed-schema and semantic validation path. Tests never make network requests.

The Codex adapter sends sanitized linguistic blueprints in batches, parses content only from the
controlled last-message file, and recomputes canonical spans locally. Its JSONL event stream may
contribute token and duration metadata but is never treated as generated record content. See
`docs/policybench_codex_generation.md` for its security and resume contract.

## Response contract

The provider returns natural-language context and candidate fields plus exact substring anchors. It
does not return canonical offsets or choose taxonomy labels. Unknown
keys, malformed JSON, missing anchors, undeclared authority claims, wrong languages, label leakage,
changed actions/sources, and oversized values are rejected.

Candidate text is inert data. The system prompt explicitly forbids following it, executing it,
browsing on its behalf, or calling a described tool.

## Validation and retry

Each attempt follows this order:

1. Enforce response byte and JSON limits.
2. Validate the closed generation-response schema.
3. Compare language, action, entities, presentation, source, and authority behavior with
   deterministic blueprint constraints.
4. Reject canonical label names leaked into candidate text.
5. Resolve each substring anchor against the exact candidate with an explicit occurrence index.
6. Validate span bounds, exact substring equality, required types, and allowed overlaps.
7. Construct the canonical record and validate the PolicyBench record profile and derived verdict.
8. Run duplicate checks according to configuration.

Failure is never silently accepted. The pipeline stores a hash, attempt number, validation codes,
and bounded diagnostic text for every rejection, then retries up to `max_retries`. Exhaustion leaves
the scenario rejected or excluded; it never fabricates missing information.

The aggregate quality report publishes acceptance, rejection, and retry rates; rejection reason and
attempt counts; and span-specific failure checks, rejected attempts, and span-rejection rate. It also
reports coverage, length percentiles, duplicate statistics, and cost coverage without reproducing
candidate text.

## Artifacts and resume

Raw responses are local under `data/generated/raw/` and remain ignored by Git. Accepted immutable
artifacts contain the exact raw response text, its UTF-8 SHA-256 binding,
blueprint/policy/prompt/configuration hashes, generation metadata, validation evidence, and an
artifact self-checksum. Canonical releases are built separately under
`data/generated/policybench-v0.1/`.

Resume verifies the accepted artifact and existing canonical record hashes before reuse. A validated
scenario is not regenerated unless explicitly forced. Corrupt, mismatched, or configuration-drifted
state is rejected. Concurrency does not affect bytes because results are sorted by scenario ID before
serialization.

## Offline rebuild

An offline rebuild performs no provider acquisition, including no mock-provider call. It consumes
accepted artifacts, reruns canonicalization and validation, and writes stable JSON with LF endings,
UTF-8, sorted keys, compact JSONL, stable file ordering, and checksum manifests. Given identical
inputs, it is byte-identical.
If any selected accepted artifact is missing or fails its blueprint, policy, prompt, configuration,
raw-response, or self-checksum binding, offline mode fails rather than acquiring a replacement.

Remote providers may be nondeterministic even with a seed. Reproducibility therefore concerns the
immutable accepted response, not a promise that a remote API call can be repeated byte-for-byte.
