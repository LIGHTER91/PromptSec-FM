# Phase 3.3 — InjecAgent and AgentDojo integration

Build: `promptsec-dataset-v0.2`, generated 2026-07-12 from the pinned local caches.
No model, benchmark attack, defense, tool execution, API key, or runtime result was used.

## Pinned inputs and observed counts

| Source | Immutable pin | Acquisition | Imported | Skipped |
|---|---|---|---:|---:|
| PromptInject | `2928a719d5de62d3766226f1b44c51d9570bc530` | pinned repository artifact | 10 | 0 |
| BIPIA | `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` | pinned repository artifacts | 250 | 0 |
| Open-Prompt-Injection | `95290f7ce3794c4c52ad3fe8113db2bfcdfe89e0` | pinned repository archive | 28 | 0 |
| NotInject | `1b5751e88bf7475acbedfc8eda795ce060307c84` | pinned source files | 339 | 0 |
| InjecAgent | `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` | seven pinned raw files | 2,108 | 0 |
| AgentDojo | package `0.1.35`, Git `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b`, benchmark `v1.2.2` | pinned wheel plus static API snapshot | 949 | 0 |

The v0.2 import total is **3,684**. The AgentDojo snapshot contains four suites,
97 user tasks, 35 injection tasks, and 949 user-task/injection-task pairs. Its snapshot
SHA-256 is `2edb1a68f94ed3a24323c4f7b9e94797a6856f8083e7d4b0faf9daec7e955fc9`.

## Canonical mapping

Both new importers use the existing `BaseImporter` contract and the frozen Taxonomy v1.0.
Mappings are source-derived and marked `NEEDS_REVIEW`; they are never human gold.

- InjecAgent preserves user/attacker/test-case identifiers, tools, attack category,
  direct-harm versus data-stealing setting, base/enhanced variant, S1/S2 metadata where
  explicitly present, exact payload boundaries, and the complete upstream Tool Response
  checksum. Data-stealing cases use the explicit `SENSITIVE_DATA_EXFILTRATION` and
  `UNAUTHORIZED_ACTION` objectives. No authority spoofing or policy conflict is inferred.
- AgentDojo uses one static record per suite/user-task/injection-task pair. It copies the
  public task prompt and injection goal definition, suite tools, task difficulties, source
  classes/modules, state/vector references, and benchmark versions. It does not execute
  `get_injection_candidates`, task ground truth, tools, attacks, or defenses. Objectives,
  authority, and alignments remain unresolved.

The mapping-confidence distribution is: 10 records at 1.00, 250 at 0.70, 28 at 0.80,
339 at 0.20, 2,108 at 0.80, and 949 at 0.60. The release review queue contains all
3,684 records; 3,674 require manual review according to the quality profile, and every
record is a review candidate because the threshold is 0.85 or an uncertainty axis is set.

## Quality, deduplication, and splits

- Exact duplicate groups: 2; exact duplicates dropped: 2.
- Semantic clusters: 491; cross-source semantic clusters: 0 in this build.
- Materialized records: 3,682.
- Splits: train 211, validation 82, test_id 300, held-out source 28,
  held-out family 4, and `test_agentic_provisional` 3,057.
- InjecAgent base/enhanced parent groups and AgentDojo task-pair groups are atomic.
  Agentic records are excluded from training and from gold evaluation claims; their
  dedicated partition is explicitly provisional.

The complete aggregate audit is in `reports/data_statistics/dataset_v0.2.json` and
`dataset_v0.2.md`. It includes source/language/domain distributions, families,
objectives, delivery modes, mapping tiers/confidence, review rates, unknown axes,
content and span lengths, benchmark suites, tools, and source-specific agentic metadata.

## Licensing and reproducibility

Repository code and redacted metadata are publishable. BIPIA remains
`BLOCKED_PENDING_LICENSE_REVIEW`, and InjecAgent/AgentDojo task payloads remain
`NOASSERTION`/local reconstruction only. Generated JSONL payloads, review queues, full
checksums, caches, snapshots, and model outputs are ignored by Git; only safe release
metadata is tracked.

Rebuild after acquisition:

```powershell
python scripts/build_dataset.py --config configs/dataset_v0.2.yaml `
  --output data/releases/promptsec-dataset-v0.2 --offline
```

The first completed v0.2 build produced full local checksum manifest
`839e30067b643b4e605a776cb15033820d05a1a4a883e7ac5db4d0a7fa97a712`.
Two consecutive offline builds must be compared before the Phase 3.3 commit; no
payload is redistributed by that commit.

Known limitations: AgentDojo records represent static task definitions rather than
materialized injected vectors; source-derived labels require human review; and the
corpus remains strongly imbalanced toward InjecAgent and AgentDojo.
