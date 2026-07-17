# PromptSec-PolicyBench v0.1 reproducibility

## Reproducibility target

The same configuration, seed, policy catalogues, scenario blueprints, versioned prompts, and accepted
raw responses produce byte-identical canonical records, reports, split files, and manifests.
Acquiring a new response from a remote provider is intentionally outside this guarantee.

## Environment

Use Python 3.11 or later and install the repository with development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Provider credentials are supplied through the environment-variable name configured in the YAML.
Never place an API key in the repository, command history, prompts, reports, or accepted artifacts.

## Catalogue validation

```bash
python scripts/validate_policy_catalog.py \
  --policies data/policybench/policies
```

Validation is local and requires no provider or network access.

## Network-free mock acquisition

Build a small deterministic release and its accepted artifacts without contacting a network:

```bash
python scripts/generate_policybench.py \
  --config configs/policybench_v0.1.yaml \
  --output data/generated/policybench-v0.1 \
  --provider mock \
  --max-records 24 \
  --seed 20260715
```

This is an acquisition even though the provider is deterministic and local. It writes raw responses
and accepted artifacts. Do not add `--offline` to a fresh build: `--offline` is
accepted-artifact-only replay and rejects a missing artifact rather than invoking a provider.

The default configuration targets 6,000 records. The full generated dataset, raw outputs, accepted
artifacts, caches, and review packets are ignored by Git.

## Validation and reporting

```bash
python scripts/validate_policybench.py data/generated/policybench-v0.1
python scripts/report_policybench.py data/generated/policybench-v0.1 \
  --output reports/policybench-v0.1
```

The validator checks every canonical record, checksum, counterfactual group, qualified split-leakage
invariant, and manifest. The split report separately exposes raw cross-split identifiers for
language-stratified families; counterfactual groups and semantic duplicate clusters remain globally
atomic. The quality report contains aggregate distributions, acceptance/rejection/retry rates,
span-rejection diagnostics, and hashes rather than full candidate text.

For linkage, policy, scenario, attack, and base-generation families are language-stratified; their
raw cross-split IDs are not claimed disjoint. The policy-family OOD view additionally requires its
selected raw policy families to be absent from train. Full split construction rejects empty or
underpowered requested splits and record-target deviations larger than one component.

## Review-packet smoke test

```bash
python scripts/create_policybench_review_packets.py \
  --dataset data/generated/policybench-v0.1 \
  --output data/review/policybench-gold-candidate-v0.1 \
  --records 12 \
  --seed 20260715
```

Packet creation does not create or promote gold annotations.

## Offline rebuild from accepted artifacts

After an acquisition run, retain the immutable accepted-artifact directory and its checksum
manifest. Rebuild the same selection with:

```bash
python scripts/generate_policybench.py \
  --config configs/policybench_v0.1.yaml \
  --output data/generated/policybench-v0.1-replay \
  --offline --resume \
  --max-records 24 \
  --seed 20260715
```

The command refuses a raw-response, self-checksum, blueprint, policy, prompt, or configuration
mismatch and never invokes a provider to replace a missing or invalid artifact. Compare
`checksums.sha256` and the release-manifest hash across builds.

`examples/policybench/` contains a tiny two-record mock-provider replay fixture, with one French
scenario expanded into its blueprint, closed generation response, and canonical record. Its README
shows how to verify both accepted artifacts with `--offline --max-records 2` without a network
request.

## Serialization rules

- UTF-8 with strict decoding and LF endings;
- no Unicode normalization of canonical candidate text;
- Python Unicode code-point offsets and half-open spans;
- sorted record IDs and file names;
- sorted JSON keys, compact JSONL, and a final newline;
- no NaN or non-standard JSON constants;
- fixed acquisition timestamps preserved from accepted artifacts; and
- SHA-256 over exact bytes or canonical JSON as documented.

Deduplication may normalize a copy with NFKC, case folding, and whitespace collapse for comparison.
It never mutates canonical text or offsets.

## Verification gates

Before declaring a local release ready for human review, run:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The test suite uses the deterministic mock provider and makes no network requests. Record the Python
version, repository revision, config hash, policy hashes, prompt hashes, accepted-artifact manifest,
release checksums, commands, and any rejected-attempt counts in the experiment log.
