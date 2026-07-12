# Phase 3.4 targeted quality review

Phase state: **`READY_FOR_HUMAN_REVIEW`**.

The preflight on the current HEAD completed with 136 existing tests, Ruff and format
checks passing, v0.2 split validation passing, and an offline rebuild checksum of
`839e30067b643b4e605a776cb15033820d05a1a4a883e7ac5db4d0a7fa97a712`. The review
infrastructure adds four tests; the resulting local suite has 139 passing tests.

The complete v0.2 corpus contains 3,682 materialized records and 491 semantic clusters.
The deterministic candidate protocol selected 500 records from all six sources, with
333 agentic cases, 109 NotInject hard negatives, 28 Open-Prompt-Injection cases, 22
BIPIA cases, 8 PromptInject cases, and the remaining candidates from InjecAgent and
AgentDojo. Selection uses the fixed seed in `configs/gold_subset_v0.1.yaml`, source
coverage, explicit hard-negative/agentic minimums, one representative per semantic
cluster where possible, and stable SHA-256 ranking.

The structural audit found 258 alignments without a usable user goal, 159 contextual
hash conflicts, and 278 source-derived records marked `CONFIRMED`; these are reported,
not automatically repaired. Priority bands and machine-readable reasons are in
`review_priority.jsonl` and `review_priority_summary.md`.

Two blinded packets are generated with identical candidate IDs in independent deterministic
orders. They contain only taxonomy-relevant content/context and empty annotation fields;
source IDs, original labels, automatic mappings, clusters, priorities and license status
are kept in the researcher-only manifest. The packet directory is ignored by Git.

No human annotation files or adjudication file are present. Therefore no gold subset is
claimed, no labels are promoted, and no `GOLD_LOCAL_ONLY` or `GOLD_REDISTRIBUTABLE` state
is asserted. `scripts/adjudicate_gold.py` reports the next state only after two complete
human annotation files exist.
