# PromptSec-Dataset v0.1

PromptSec-Dataset v0.1 is an **experimental audited release**. It is not a final training split, and this build does not train a model.

## Scope

- Frozen normative taxonomy: PromptSec-FM v1.0
- Imported source records: 627
- Materialized records after exact deduplication: 625
- Semantic clusters: 399
- Review-queue records: 627

## Source distribution before deduplication

| Source | Records |
|---|---:|
| bipia | 250 |
| notinject | 339 |
| open_prompt_injection | 28 |
| promptinject | 10 |

## Mapping evidence

Mapping tiers describe the evidence used to migrate source labels. They must not be treated as equivalent to human annotation.

| Tier | Records |
|---|---:|
| DETERMINISTIC_MAPPING | 10 |
| GOLD_SOURCE | 0 |
| HEURISTIC_MAPPING | 278 |
| UNANNOTATED | 339 |

## Experimental splits

| Split | Records |
|---|---:|
| train | 237 |
| validation | 85 |
| test_id | 271 |
| test_held_out_source | 28 |
| test_held_out_family | 4 |

NotInject is weighted toward validation and `test_id` to measure over-defense. Source `open_prompt_injection` is reserved for `test_held_out_source`; family `prompt_or_policy_disclosure` is reserved for `test_held_out_family`.

Leakage checks:

- No semantic cluster crosses materialized splits: True
- Held-out source absent from train: True
- Held-out family absent from train: True
- No template family overlaps held-out-family test and train: True
- Exact duplicates excluded: True

## Deduplication and grouping

Exact grouping uses raw, normalized, and context-aware SHA-256 hashes. Semantic clusters use a deterministic lexical similarity rule with documented synonym normalization; no learned embedding or model is used. Paraphrases are retained as variants or sent to review, and clusters remain atomic across splits.

Template families come only from source metadata, generation templates, documented mapping rules, or manual review. See `docs/family_mapping_v0.1.md`.

## Limitations and licensing

The 627 imported records are strongly source-imbalanced. PromptInject and Open-Prompt-Injection are small, while BIPIA and NotInject dominate. The release contains uncertain and non-annotated mappings explicitly isolated in the review queue. Consult `licenses.json` before redistribution: obligations are component-specific.

Publication status: **BLOCKED_PENDING_LICENSE_REVIEW**. BIPIA attack-template redistribution is recorded as NOASSERTION/unknown, so this local experimental artifact must not be published until that license review is resolved.

## Rebuild

```bash
python scripts/build_dataset.py --config configs/dataset_v0.1.yaml --output data/releases/promptsec-dataset-v0.1
```

`checksums.sha256` covers every other release file. With the pinned artifacts and configuration, two builds are byte-identical.
