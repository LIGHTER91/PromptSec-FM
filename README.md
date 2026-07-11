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

## Mapping contract

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
data/                 Local raw data and the tracked v0.1 release
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
