# PromptSec-Dataset v0.2 family mapping

This document extends the Phase 3.2 grouping contract in
`docs/family_mapping_v0.1.md` for the InjecAgent and AgentDojo integrations. The
rules are experimental leakage-control metadata. They do not add to, rename, or
reinterpret any PromptSec-FM Taxonomy v1.0 label.

## Evidence and review contract

The v0.1 evidence precedence remains unchanged: explicit source metadata, upstream
generation structure, a documented source-specific rule, then manual review. The
grouping implementation never inspects `content.text` to assign a
`template_family`. Missing relationship metadata produces the `manual_review`
family and a review requirement instead of a keyword-derived guess.

Source-derived grouping is not human annotation. In particular, sharing a source
template says nothing by itself about authority, user-goal alignment, protected
policy alignment, or attack success.

## Agentic source rules

| Source evidence | `domain` | `template_family` | Rule basis |
|---|---|---|---|
| InjecAgent `attack_mode`, `attacker_case_id`, and `attack_category` | Upstream attack category | `injecagent_<attack_mode>_<attacker_case_id>` | The pinned benchmark explicitly joins an attacker case to generated test cases. The family is not inferred from attack text. |
| AgentDojo `suite_id` and `injection_task_id` | Upstream suite | `agentdojo_<suite_id>_<injection_task_id>` | The pinned static suite definition explicitly associates the injection task with the suite. No benchmark run is required. |

InjecAgent `base` and `enhanced` settings are variants rather than separate
families. The hacking prefix remains provenance metadata and does not create a new
taxonomy value. AgentDojo user-task/injection-task pairs are canonical security-case
definitions, while the suite/injection-task key is the broader family used for
leakage control.

## Semantic and parent grouping

The semantic deduplication stage receives structured source keys in addition to its
cross-source lexical comparison:

- InjecAgent uses `(attack_mode, attacker_case_id)`. This keeps every user-case
  realization of one attacker template, including base and enhanced variants, in a
  single semantic cluster.
- AgentDojo uses `(suite_id, injection_task_id)`. This keeps the same injection-task
  definition together across user tasks in that suite.

InjecAgent also records a parent identifier derived from
`(attack_mode, attacker_case_id, user_case_id)`, so the base/enhanced pair cannot be
split even if later clustering rules change. AgentDojo records a stable security-case
identifier derived from `(benchmark_version, suite_id, user_task_id,
injection_task_id)`. Split validation checks these relationship keys independently
of surface similarity.

Exact duplicate removal retains the provenance of every contributing source. A
structured group can still join an existing lexical-semantic cluster from another
source, which prevents equivalent older and agentic examples from crossing
incompatible splits.

## Provisional split policy

For v0.2, InjecAgent and AgentDojo are configured as agentic sources. Their kept
records are assigned cluster-atomically to `test_agentic_provisional`; uncertain
mappings are also emitted to the agentic review queue. This partition is not a gold
test set and must not be reported as human-verified performance data.

The v0.1 held-out-source, held-out-family, and NotInject hard-negative policies stay
in force for the earlier sources. Agentic grouping takes precedence so that an
agentic cluster cannot leak into training, validation, ID test, or either established
holdout.
