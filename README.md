# PromptSec-FM

PromptSec-FM is a research project for building a reproducible, provenance-aware
dataset about prompt injection and related prompt-mediated attacks.

## Normative taxonomy

PromptSec-FM Taxonomy v1.0 is frozen. Its normative sources are:

- `docs/taxonomy.md`
- `docs/annotation_guidelines.md`
- `docs/taxonomy_migration_v1.md`
- `schemas/promptsec-annotation-v1.schema.json`
- `examples/promptsec-annotation-example-v1.json`

The taxonomy keeps instruction properties, goal and policy alignment, authority,
attack objectives, and content provenance on separate axes. Legacy labels such as
`SAFE`, `MISALIGNED_INSTRUCTION`, and `trust_level` are not canonical labels.

The literature matrix remains in `docs/Matrice_litterature_PromptSec-FM.xlsx`.
Superseded literature material is kept in `docs/archive/` for scientific traceability
and is not a normative specification.

## Dataset pipeline

Phase 3 adds PromptSec-Dataset v0.1: a reproducible pipeline that imports public
source datasets, retains their original records and labels, maps them conservatively
to Taxonomy v1.0, validates canonical records, and records source and license
provenance.

The first supported sources are PromptInject, BIPIA, Open-Prompt-Injection, and
NotInject.

| Source | Initial v0.1 scope | Records |
|---|---|---:|
| PromptInject | Safely extracts the two attack-template dictionaries with `ast.literal_eval`; upstream Python is never executed. | 10 |
| BIPIA | Imports the four text/code attack-template JSON files. Licensed context datasets remain separate. | 250 |
| Open-Prompt-Injection | Reads the pinned source ZIP in memory and pairs explicitly configured injected-task variants with target-task templates; downstream datasets are not redistributed. | 28 |
| NotInject | Imports the three official 113-record JSON mirrors as benign hard negatives without inventing a `SAFE` label. | 339 |

## Quick start

PromptSec-Dataset requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

Fetch artifacts declared by one pinned source configuration:

```bash
python scripts/fetch_sources.py \
  --config configs/sources/promptinject.toml \
  --destination data/raw
```

The command prints an `ARTIFACT_ID -> local path` mapping. Pass that mapping to the
builder; repeat `--input` for sources with several artifacts:

```bash
python scripts/build_dataset.py \
  --config configs/sources/promptinject.toml \
  --input prompt_data=data/raw/promptinject/2928a719d5de62d3766226f1b44c51d9570bc530/prompt_data.py \
  --output data/processed/v0.1/promptinject.jsonl \
  --report reports/promptinject.json \
  --imported-at 2026-07-11T00:00:00Z

python scripts/validate_dataset.py data/processed/v0.1/promptinject.jsonl
```

`--imported-at` makes build metadata reproducible across runs. Raw artifacts and
intermediate records stay ignored by Git. Their expected SHA-256 values, upstream
revisions, and component-level license notes are tracked in `configs/` and
`manifests/`.

## Audited experimental release

Phase 3.2 audits mappings, computes exact and lexical-semantic duplicate groups,
assigns provenance-backed template families, and creates leakage-checked
experimental splits. It does not train a model. Build the complete release with one
command:

```bash
python scripts/build_dataset.py \
  --config configs/dataset_v0.1.yaml \
  --output data/releases/promptsec-dataset-v0.1
```

The command fetches missing pinned artifacts, validates all canonical records, and
writes the five split files, review queue, dataset card, source and license
inventories, statistics, deduplication and split reports, and SHA-256 checksums.
The same configuration and source artifacts produce byte-identical release files.
The family mapping rules are documented in `docs/family_mapping_v0.1.md`.

The generated v0.1 artifact is local and experimental. Its publication status is
`BLOCKED_PENDING_LICENSE_REVIEW` because redistribution permission for BIPIA's
attack-template JSON is unresolved; consult `licenses.json` before any publication.

## Agentic source integration

Phase 3.3 extends the same pipeline with InjecAgent and AgentDojo. It does not train
or execute a model, an attack, a defense, or a benchmark. Both integrations consume
static definitions from immutable upstream pins:

| Source | Immutable pin | Static import unit |
|---|---|---|
| InjecAgent | Git commit `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` | One base or enhanced test-case variant, linked to its user case and attacker case. |
| AgentDojo | Package `agentdojo==0.1.35`, Git commit `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, benchmark request `v1.2.2` | One static user-task/injection-task pair from a deterministic snapshot of the public suite API. |

AgentDojo snapshot export is an acquisition-time operation, not part of an ordinary
offline build. Install its exact optional dependency when preparing that cache:

```bash
python -m pip install -e ".[dev,agentdojo-snapshot]"
python scripts/download_sources.py --source injecagent
python scripts/download_sources.py --source agentdojo
```

`--source` is repeatable, and `--all` prepares all configured sources. A prepared
local checkout, artifact, or cache can be supplied without changing a pin:

```bash
python scripts/download_sources.py \
  --source injecagent \
  --local-path injecagent=/path/to/InjecAgent
```

Acquisition verifies the configured revision or checksum. A checkout at another
commit is rejected; the command never silently follows a moving branch. Once the
cache is prepared, build PromptSec-Dataset v0.2 without network access:

```bash
python scripts/build_dataset.py \
  --config configs/dataset_v0.2.yaml \
  --output data/releases/promptsec-dataset-v0.2 \
  --offline
```

`--source-path SOURCE=PATH` provides the equivalent per-source override to the
release builder. Validate locally materialized files and run the quality gates with:

```bash
python scripts/validate_dataset.py \
  data/releases/promptsec-dataset-v0.2/test_agentic_provisional.jsonl
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Agentic labels are source-derived mappings and require review; they are not human
gold annotations. The AgentDojo import records static task definitions, not attack
success, tool execution, model behavior, or a materialized runtime injection. Its
agentic evaluation split is therefore explicitly provisional. The repository-level
MIT licenses do not by themselves resolve every dataset or derived-payload right:
InjecAgent and AgentDojo payloads remain local under `NOASSERTION`, and the existing
BIPIA publication block remains in force. Git tracks only reconstructible code,
manifests, reports without source text, and release metadata—not generated JSONL,
source caches, snapshots, state dumps, embeddings, or model outputs.

## Mapping contract

## Phase 3.4 human-review preparation

Phase 3.4 audits the v0.2 corpus, scores review priorities, selects a deterministic
approximately-500-record candidate set, and creates two blinded annotation packets.
It does not create gold labels. With no completed human annotation files, the explicit
state is `READY_FOR_HUMAN_REVIEW`.

```powershell
python scripts/create_gold_review_packets.py `
  --config configs/gold_subset_v0.1.yaml `
  --release data/releases/promptsec-dataset-v0.2 `
  --output data/review/gold-candidate-v0.1
python scripts/adjudicate_gold.py `
  --packet-dir data/review/gold-candidate-v0.1 `
  --output reports/quality_review/adjudication_state.json
```

Candidate payloads and researcher manifests remain ignored by Git. Annotators see only
taxonomy-relevant context and content; source IDs, automatic labels, mappings, clusters,
and license status remain hidden. Human double annotation and adjudication are required
before any `GOLD_LOCAL_ONLY` or redistributable state can be claimed.

- Exact upstream objects and labels are retained under
  `metadata.dataset_provenance.source_record`.
- Corpus acquisition provenance never replaces the frozen scenario-provenance axes
  in `content`.
- Text is hashed as exact UTF-8 without Unicode normalization; spans use Python
  Unicode code-point offsets and the half-open convention `[start, end)`.
- Non-deterministic or context-incomplete migrations are marked `NEEDS_REVIEW`; an
  upstream benign label does not imply `NO_INSTRUCTION`.
- Every record is checked against the new dataset-record v0.1 profile, the unchanged
  annotation schema v1.0, and semantic checks for spans, verdict derivation, fallback
  labels, and checksums.

## Repository layout

```text
configs/              Source and release configuration
data/                 Ignored raw data/payloads and tracked safe release metadata
docs/                 Normative and scientific documentation
examples/             Canonical examples
manifests/            Source and license manifests
reports/              Tracked v0.1 audit reports and ignored local reports
schemas/              JSON Schemas
scripts/              Command-line entry points
src/promptsec/data/   Dataset pipeline and importers
tests/data/           Pipeline tests and small fixtures
```

## Research status

This repository is under active research development. Archived documents and source
dataset labels must not be interpreted as PromptSec-FM ground truth unless they have
been migrated and validated against the frozen v1.0 taxonomy.
