# PolicyBench generation with Codex CLI

The `codex_cli` provider uses a locally installed Codex CLI and the user's existing ChatGPT
subscription authentication. It does not accept, read, or require an API key, and it never falls
back to another provider. The generated records remain `SILVER_VALIDATED` with
`human_validation_status=PENDING`, `annotator_confidence=0.0`, and no automatic Gold promotion.

## Prerequisites

Verify the local executable and authentication without inspecting authentication storage:

```powershell
codex --version
codex login status
codex exec --help
```

`codex login status` must report ChatGPT authentication. If it does not, stop and run `codex login`.
The model is deliberately explicit rather than inherited from personal Codex configuration:

```powershell
$env:PROMPTSEC_CODEX_MODEL = "gpt-5.6-sol"
```

Model availability is account-dependent. Use the exact model identifier available to the signed-in
account; a model-selection failure is reported and is never hidden by a fallback. The model and
Codex CLI version are bound into every accepted artifact's acquisition fingerprint.

## Batching and deterministic responsibility

`configs/policybench_codex_v0.1.yaml` reuses the complete deterministic 6,000-scenario plan and
defaults to ten records per batch, concurrency one, four bounded retries, a 900-second timeout,
strict structured output, and high reasoning effort. Supported batch sizes are 1, 2, 5, 10, and 20.
Complete two-record counterfactual units are packed into one batch and are never split.

The deterministic planner fixes taxonomy labels, expected alignments, source and authority status,
policy semantics, category, split, and Gold status before acquisition. Codex receives sanitized
blueprints and realizes only protected-policy text, legitimate user-goal text, candidate text, and
exact span substrings. Canonical Unicode offsets are recomputed locally from the returned substrings;
provider-supplied offsets are neither requested nor trusted.

The acquisition fingerprint includes provider, exact model, CLI version, batch size, reasoning
effort, configuration and catalogue hashes, every generation prompt hash, output-schema hash,
temperature-equivalent configuration, seed, and taxonomy version. Artifacts from another provider,
model, CLI, prompt, schema, or batch configuration cannot be resumed.

## Sandbox and credential boundary

Each batch is launched using a subprocess argument array with `shell=False`; the prompt is sent on
stdin and candidate text is never interpolated into a command. The provider:

- creates a fresh temporary working directory containing only the strict output schema;
- runs `codex exec` with ephemeral mode, no approval prompts, read-only sandboxing, ignored user
  config and project rules, and no repository checkout;
- explicitly requires ChatGPT login, disables web search, the shell tool, app tools, skill dependency
  installation, update checks, and history persistence;
- passes a minimal environment that excludes API-key and other secret-like variables while retaining
  only the normal OS paths needed for the CLI to reuse its saved authentication;
- enforces strict UTF-8, prompt/output byte limits, a process timeout, closed JSON Schema, exact batch
  cardinality and scenario IDs, and temporary-directory cleanup; and
- parses generated content only from the controlled `--output-last-message` file. JSONL stdout is
  used only for diagnostics and usage metadata.

The dedicated `prompts/policybench/codex_batch_v9.txt` prompt treats every candidate instruction as
inert, untrusted research data. Normal operational reporting uses scenario IDs and hashes rather
than reproducing attack payloads. Raw and accepted artifacts remain under ignored generated-data
directories.

## Pilot and validation

The `--resume` switch is the acquisition-state control; it is intentionally a CLI option rather
than a semantic plan field. A 20-record pilot uses at most two fresh ten-record model invocations
when no matching artifact is already accepted:

```powershell
& .\.venv\Scripts\python.exe scripts\generate_policybench.py `
  --config configs\policybench_codex_v0.1.yaml `
  --output data\generated\policybench-codex-v0.1-pilot `
  --provider codex_cli `
  --model $env:PROMPTSEC_CODEX_MODEL `
  --seed 20260715 `
  --max-records 20 `
  --max-retries 4 `
  --concurrency 1 `
  --temperature 0.0 `
  --resume

& .\.venv\Scripts\python.exe scripts\validate_policybench.py `
  data\generated\policybench-codex-v0.1-pilot

& .\.venv\Scripts\python.exe scripts\report_policybench.py `
  data\generated\policybench-codex-v0.1-pilot `
  --output reports\policybench-codex-v0.1-pilot
```

Usage metadata records input, cached input, output, and reasoning output tokens when Codex emits
them, plus batch duration and exit status. ChatGPT subscription activity is reported as
`authentication_mode=chatgpt_subscription`, `api_cost=not_applicable`, and
`account_usage_limits=externally_managed`; API monetary cost is not inferred.

Rate/usage limits and transient failures use bounded retries. Every already validated batch is
checkpointed before the next invocation, so the same command with `--resume` continues from those
artifacts. The provider never changes provider, model, batch size, or record target silently.

## Later full acquisition

Do not add `--max-records` when a later, explicitly authorized 6,000-record acquisition is desired:

```powershell
$env:PROMPTSEC_CODEX_MODEL = "gpt-5.6-sol"
& .\.venv\Scripts\python.exe scripts\generate_policybench.py `
  --config configs\policybench_codex_v0.1.yaml `
  --output data\generated\policybench-codex-v0.1 `
  --provider codex_cli `
  --model $env:PROMPTSEC_CODEX_MODEL `
  --seed 20260715 `
  --max-retries 4 `
  --concurrency 1 `
  --temperature 0.0 `
  --resume
```

At ten records per batch, a complete run requires about 600 successful batch invocations, plus
bounded retry invocations. Duration must be estimated from measured pilot batches because ChatGPT
account limits and service latency are externally managed.
