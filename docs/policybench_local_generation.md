# Local PolicyBench generation

PolicyBench can acquire candidates from a locally managed OpenAI-compatible Chat Completions
server. Local generation uses `configs/policybench_local_v0.1.yaml`, keeps concurrency at one, and
writes to local-specific raw, accepted, release, report, and review paths. It does not fall back to
the mock provider.

## Security and compatibility

`authentication: optional_for_loopback` permits a missing API key only when URL parsing proves the
hostname is `localhost` or an IP address in the loopback range. Plain HTTP is also limited to those
hosts. User-info, queries, fragments, deceptive suffixes, and non-loopback HTTP endpoints are
rejected. HTTPS endpoints outside loopback still require a key. When the key variable is absent, no
`Authorization` header is sent; a token may be supplied in the current shell if a local server was
configured to require one. Secrets are never part of prompts, artifacts, or cache fingerprints.

The `response_mode` setting supports:

- `strict_json_schema`: send the closed schema using strict Chat Completions JSON Schema;
- `json_object`: request JSON mode and place the same closed schema in the untrusted-data prompt;
- `prompt_constrained_json`: omit `response_format` and constrain output in the prompt.

All modes use the same strict UTF-8/JSON parser, reject surrounding prose and unknown fields, and
run every semantic, span, counterfactual, duplicate, and record validator. The local example uses
`json_object` for broad server compatibility. Switch to `strict_json_schema` only after confirming
the installed server/model supports it.

Acquisition fingerprints bind the provider, model, model revision, endpoint, authentication policy,
response mode, temperature, seed, prompts, policy catalogues, and full configuration hash. Changing
any of those request inputs makes an existing accepted artifact ineligible for resume. Credential
values are deliberately excluded.

## Recommended Windows setup for this host

This host has about 8 GiB RAM, an Intel Core i5-1135G7, and integrated Intel Iris Xe graphics with
no CUDA device. Use Ollama with `qwen2.5:3b-instruct-q4_K_M`: the published artifact is about 1.9 GB
and the model has 3.09B parameters. Allow additional disk for the Ollama installation and runtime
files and several GiB of available system memory during inference. Close memory-heavy applications
before a run. The 3B model uses the Qwen Research License; review it for the intended dataset use.

Run these steps manually in a new PowerShell window:

```powershell
# Install the official Windows build, then reopen PowerShell if PATH has not refreshed.
irm https://ollama.com/install.ps1 | iex

ollama pull qwen2.5:3b-instruct-q4_K_M
ollama serve
```

The Windows desktop installation normally starts Ollama in the background. If `ollama serve`
reports that the address is already in use, keep the existing background server and do not start a
second instance. The OpenAI-compatible base URL is `http://127.0.0.1:11434/v1`.

In the PromptSec repository, use a separate PowerShell window:

```powershell
$env:PROMPTSEC_GENERATION_PROVIDER = "openai_compatible"
$env:PROMPTSEC_GENERATION_MODEL = "qwen2.5:3b-instruct-q4_K_M"
$env:PROMPTSEC_GENERATION_BASE_URL = "http://127.0.0.1:11434/v1"
Remove-Item Env:PROMPTSEC_GENERATION_API_KEY -ErrorAction SilentlyContinue

# Connectivity smoke test: exactly 20 selected records.
& .\.venv\Scripts\python.exe scripts\generate_policybench.py `
  --config configs\policybench_local_v0.1.yaml `
  --output data\generated\policybench-local-v0.1-smoke `
  --provider openai_compatible `
  --model $env:PROMPTSEC_GENERATION_MODEL `
  --seed 20260715 `
  --max-records 20 `
  --max-retries 4 `
  --concurrency 1 `
  --temperature 0.7 `
  --resume

# Only after the smoke test validates: exactly 100 selected records.
& .\.venv\Scripts\python.exe scripts\generate_policybench.py `
  --config configs\policybench_local_v0.1.yaml `
  --output data\generated\policybench-local-v0.1-pilot `
  --provider openai_compatible `
  --model $env:PROMPTSEC_GENERATION_MODEL `
  --seed 20260715 `
  --max-records 100 `
  --max-retries 4 `
  --concurrency 1 `
  --temperature 0.7 `
  --resume

& .\.venv\Scripts\python.exe scripts\validate_policybench.py `
  data\generated\policybench-local-v0.1-pilot
& .\.venv\Scripts\python.exe scripts\report_policybench.py `
  data\generated\policybench-local-v0.1-pilot `
  --output reports\policybench-local-v0.1-pilot
```

Do not run smoke and pilot concurrently because both use the same accepted-artifact directory.
The distinct release outputs prevent a partial smoke release from being mistaken for the pilot.
Estimate the full 6,000-record duration only after the real pilot: `pilot elapsed seconds * 60`,
then widen the estimate for observed retry rate and thermal throttling.

## LM Studio alternative

LM Studio also exposes OpenAI-compatible endpoints on `http://127.0.0.1:1234/v1` and does not
require authentication by default. After installing and launching LM Studio manually, its `lms`
CLI is available. Download a suitable 3B Q4 model through the application or `lms get`, then use the
exact model key reported by `lms ls`:

```powershell
lms ls
lms load <MODEL_KEY> --identifier policybench-local --context-length 8192 --gpu off
lms server start --bind 127.0.0.1 --port 1234

$env:PROMPTSEC_GENERATION_MODEL = "policybench-local"
$env:PROMPTSEC_GENERATION_BASE_URL = "http://127.0.0.1:1234/v1"
```

Then run the same smoke and pilot commands. If LM Studio authentication is enabled, set
`PROMPTSEC_GENERATION_API_KEY` only in that shell. Never place it in YAML or `.env.example`.
