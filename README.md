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
NotInject. Pipeline code and usage are documented as the implementation is added.

## Repository layout

```text
configs/              Source configuration
data/                 Local downloaded and generated data (not committed)
docs/                 Normative and scientific documentation
examples/             Canonical examples
manifests/            Source and license manifests
reports/              Generated validation reports (not committed)
schemas/              JSON Schemas
scripts/              Command-line entry points
src/promptsec/data/   Dataset pipeline and importers
tests/data/           Pipeline tests and small fixtures
```

## Research status

This repository is under active research development. Archived documents and source
dataset labels must not be interpreted as PromptSec-FM ground truth unless they have
been migrated and validated against the frozen v1.0 taxonomy.
