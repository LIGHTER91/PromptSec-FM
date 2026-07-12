# PromptSec-Dataset v0.2

PromptSec-Dataset v0.2 is an **experimental audited release**. It is not a final training split, and this build does not train a model.

## Scope

- Frozen normative taxonomy: PromptSec-FM v1.0
- Imported source records: 3684
- Materialized records after exact deduplication: 3682
- Semantic clusters: 491
- Review-queue records: 3684

## Source distribution before deduplication

| Source | Records | Pinned revision |
|---|---:|---|
| agentdojo | 949 | a75aba7631d3ca5fb7ab938965c97ead2f9ff84b |
| bipia | 250 | a004b69ec0dd446e0afd461d98cb5e96e120a5d0 |
| injecagent | 2108 | f19c9f2c79a41046eb13c03c51a24c567a8ffa07 |
| notinject | 339 | 1b5751e88bf7475acbedfc8eda795ce060307c84 |
| open_prompt_injection | 28 | 95290f7ce3794c4c52ad3fe8113db2bfcdfe89e0 |
| promptinject | 10 | 2928a719d5de62d3766226f1b44c51d9570bc530 |

## Mapping evidence

Mapping tiers describe the evidence used to migrate source labels. They must not be treated as equivalent to human annotation.

| Tier | Records |
|---|---:|
| DETERMINISTIC_MAPPING | 10 |
| GOLD_SOURCE | 0 |
| HEURISTIC_MAPPING | 3335 |
| UNANNOTATED | 339 |

## Experimental splits

| Split | Records |
|---|---:|
| train | 211 |
| validation | 82 |
| test_id | 300 |
| test_held_out_source | 28 |
| test_held_out_family | 4 |
| test_agentic_provisional | 3057 |

NotInject is weighted toward validation and `test_id` to measure over-defense. Source `open_prompt_injection` is reserved for `test_held_out_source`; family `prompt_or_policy_disclosure` is reserved for `test_held_out_family`.

Leakage checks:

- No semantic cluster crosses materialized splits: True
- Held-out source absent from train: True
- Held-out family absent from train: True
- No template family overlaps held-out-family test and train: True
- Exact duplicates excluded: True

## Deduplication and grouping

Exact grouping uses raw, normalized, and context-aware SHA-256 hashes. Semantic clusters use a deterministic lexical similarity rule with documented synonym normalization; no learned embedding or model is used. Paraphrases are retained as variants or sent to review, and clusters remain atomic across splits.

Template families come only from source metadata, generation templates, documented mapping rules, or manual review. See the versioned family-mapping document in `docs/`.

## Limitations and licensing

The corpus is source-imbalanced and contains uncertain or non-annotated mappings explicitly isolated in review queues. Consult `statistics.json` for exact source shares and `licenses.json` before redistribution; obligations are component-specific.

Generated split and review-queue JSONL files are local reconstruction outputs. They are deliberately ignored by Git and are not redistributed in this release. The committed release surface contains only redacted statistics, provenance identifiers, decisions, and checksums.

Dataset-payload publication status: **BLOCKED_PENDING_LICENSE_REVIEW**. One or more source payload components, including the existing BIPIA restriction, remain NOASSERTION/unknown. Repository code and redacted reports remain separately classified in `licenses.json`.

## Rebuild

```bash
python scripts/build_dataset.py --config configs/dataset_v0.2.yaml --output data/releases/promptsec-dataset-v0.2 --offline
```

Local `checksums.sha256` covers every other release file, including ignored payloads. Tracked `checksums_pipeline.txt` covers only publishable redacted metadata. With the pinned artifacts and configuration, two builds are byte-identical. No model is trained by this command.


## Agentic provisional evaluation

InjecAgent and AgentDojo labels are source-derived definitions, not human gold annotations. Their records are isolated in `test_agentic_provisional` and `agentic_review_queue.jsonl`; none enter training. Runtime attack success, model behavior, defenses, and tool executions are not imported.

- Agentic sources outside the provisional split: False
- Agentic parent-group leakage: False