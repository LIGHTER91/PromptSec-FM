# Phase 3.4 license evidence review

Status: `READY_FOR_HUMAN_REVIEW` for annotation and `BLOCKED_PENDING_REVIEW` for
redistribution of unresolved payload components.

This document is a technical evidence audit, not legal advice. It records only explicit
upstream evidence at the pinned revisions; a repository code license is not treated as a
data-payload license.

| Source | Pin | Code | Payload | Raw/transformed payload | Metadata | Local rebuild |
|---|---|---|---|---|---|---|
| PromptInject | `2928a719d5de62d3766226f1b44c51d9570bc530` | MIT | REDISTRIBUTABLE_WITH_ATTRIBUTION | REDISTRIBUTABLE_WITH_ATTRIBUTION | REDISTRIBUTABLE | allowed |
| BIPIA | `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` | MIT | mixed: MIT, CC-BY-SA-4.0, NOASSERTION | BLOCKED_PENDING_REVIEW | REDISTRIBUTABLE | allowed |
| Open-Prompt-Injection | `95290f7ce3794c4c52ad3fe8113db2bfcdfe89e0` | MIT | NOASSERTION | LOCAL_REBUILD_ONLY | REDISTRIBUTABLE | allowed |
| NotInject/PIGuard | `1b5751e88bf7475acbedfc8eda795ce060307c84` | MIT | REDISTRIBUTABLE_WITH_ATTRIBUTION | REDISTRIBUTABLE_WITH_ATTRIBUTION | REDISTRIBUTABLE | allowed |
| InjecAgent | `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` | MIT | NOASSERTION | LOCAL_REBUILD_ONLY | REDISTRIBUTABLE | allowed |
| AgentDojo | `0.1.35`, Git `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b` | MIT | NOASSERTION | LOCAL_REBUILD_ONLY | REDISTRIBUTABLE | allowed |

Detailed machine-readable component evidence is in `license_evidence_v0.1.json` and
the source manifests under `manifests/sources/`. BIPIA remains blocked; no status was
weakened. Candidate packets are ignored local review artifacts and are not a release.

No maintainer was contacted and no issue was opened. Before any gold redistribution,
obtain explicit answers to the questions in `docs/license_questions.md`, preserve
attribution and share-alike notices where applicable, and re-run the publication gate.
