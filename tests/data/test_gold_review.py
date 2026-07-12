from __future__ import annotations

import json
from pathlib import Path

from promptsec.data.gold_review import (
    audit_corpus,
    create_blinded_packets,
    review_priorities,
    select_candidates,
)


def _record(record_id: str, source: str, cluster: str) -> dict:
    return {
        "id": record_id,
        "content": {
            "text": "synthetic review content",
            "language": "en",
            "delivery_mode": "DIRECT",
            "source_role": "USER",
            "content_origin": "CHAT_MESSAGE",
            "ingestion_path": "CHAT_INPUT",
            "modality": "TEXT",
            "source_integrity": "VERIFIED",
        },
        "context": {
            "protected_policy": None,
            "user_goal": None,
            "available_capabilities": [],
        },
        "annotations": {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": "UNDETERMINED",
            "protected_policy_alignment": "UNDETERMINED",
            "authority_status": "UNKNOWN",
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": [],
            "spans": [],
            "annotation_status": "INSUFFICIENT_CONTEXT",
            "annotator_confidence": 0.0,
        },
        "derived": {"prompt_injection_verdict": "UNCERTAIN"},
        "extensions": {
            "quality_v0_1": {
                "hashes": {"semantic_cluster_id": cluster},
                "mapping_quality": {
                    "mapping_confidence": 0.5,
                    "requires_manual_review": True,
                },
            }
        },
        "metadata": {
            "dataset_provenance": {
                "source_dataset": {"id": source},
                "source_record": {"id": f"upstream-{record_id}"},
            }
        },
    }


def test_candidate_selection_is_deterministic_and_covers_sources() -> None:
    records = [
        _record("a", "promptinject", "cluster-a"),
        _record("b", "notinject", "cluster-b"),
        _record("c", "agentdojo", "cluster-c"),
        _record("d", "injecagent", "cluster-d"),
    ]
    config = {
        "release_id": "fixture",
        "seed": 1,
        "target_records": 4,
        "include_all_sources_with_at_most": 40,
        "minimum_agentic_records": 1,
        "minimum_hard_negative_records": 1,
    }
    first, report = select_candidates(records, config)
    second, _ = select_candidates(list(reversed(records)), config)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert set(report["source_distribution"]) == {
        "agentdojo",
        "injecagent",
        "notinject",
        "promptinject",
    }


def test_packets_hide_provenance_and_start_empty(tmp_path: Path) -> None:
    records = [_record("a", "promptinject", "cluster-a")]
    result = create_blinded_packets(records, tmp_path, seed=7)
    packet = json.loads((tmp_path / "annotator_A.jsonl").read_text(encoding="utf-8"))
    assert result["phase_state"] == "READY_FOR_HUMAN_REVIEW"
    assert "source" not in packet
    assert "canonical_id" not in packet
    assert packet["annotation"]["attack_families"] == []
    assert packet["annotation"]["annotation_status"] is None


def test_audit_and_priority_never_claim_gold() -> None:
    records = [_record("a", "agentdojo", "cluster-a")]
    audit = audit_corpus(records)
    assert audit["phase_state"] == "READY_FOR_HUMAN_REVIEW"
    assert audit["gold_claim_permitted"] is False
    assert review_priorities(records)[0]["priority_band"] in {
        "P0_BLOCKING_ERROR",
        "P1_GOLD_CANDIDATE",
        "P2_MAPPING_AUDIT",
        "P3_COVERAGE",
    }
