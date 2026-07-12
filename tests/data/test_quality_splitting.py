from __future__ import annotations

from typing import Any

import pytest

from promptsec.data.quality.splitting import (
    SplitConfig,
    SplitError,
    assign_splits,
)


def _record(record_id: str, source_id: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "metadata": {
            "dataset_provenance": {
                "source_dataset": {"id": source_id},
            }
        },
    }


def _agentic_record(
    record_id: str,
    source_id: str,
    *,
    parent_group_id: str,
) -> dict[str, Any]:
    record = _record(record_id, source_id)
    record["extensions"] = {
        "agentic_source": {"parent_group_id": parent_group_id},
    }
    return record


def _dedup(cluster_id: str, decision: str = "KEEP") -> dict[str, Any]:
    return {
        "semantic_cluster_id": cluster_id,
        "representative_id": cluster_id,
        "dedup_decision": decision,
    }


def _family(template_family: str, domain: str = "fixture") -> dict[str, Any]:
    return {"template_family": template_family, "domain": domain}


def test_holdout_priority_exact_drops_and_cluster_atomicity() -> None:
    records = [
        _record("cross_source_rep", "general"),
        _record("cross_source_exact", "held_source"),
        _record("source_priority", "held_source"),
        _record("family_member", "general"),
        _record("family_peer", "general"),
        _record("train_rep", "general"),
        _record("train_exact", "general"),
    ]
    dedup = {
        "cross_source_rep": _dedup("cluster_cross_source"),
        "cross_source_exact": _dedup("cluster_cross_source", "DROP_EXACT_DUPLICATE"),
        "source_priority": _dedup("cluster_source_priority"),
        "family_member": _dedup("cluster_family"),
        "family_peer": _dedup("cluster_family"),
        "train_rep": _dedup("cluster_train"),
        "train_exact": _dedup("cluster_train", "DROP_EXACT_DUPLICATE"),
    }
    families = {
        "cross_source_rep": _family("ordinary"),
        "cross_source_exact": _family("ordinary"),
        # Source priority must win when the same record also has the held family.
        "source_priority": _family("held_family"),
        "family_member": _family("held_family"),
        "family_peer": _family("ordinary"),
        "train_rep": _family("ordinary"),
        "train_exact": _family("ordinary"),
    }
    config = SplitConfig(
        seed=17,
        held_out_source="held_source",
        held_out_family="held_family",
        general_ratios={"train": 1.0, "validation": 0.0, "test_id": 0.0},
        notinject_ratios={"train": 0.0, "validation": 0.0, "test_id": 1.0},
    )

    result = assign_splits(records, dedup, families, config)

    assert result.assignments["cross_source_exact"] == "DROPPED_EXACT"
    assert result.assignments["train_exact"] == "DROPPED_EXACT"
    assert result.assignments["cross_source_rep"] == "test_held_out_source"
    assert result.assignments["source_priority"] == "test_held_out_source"
    assert result.assignments["family_member"] == "test_held_out_family"
    assert result.assignments["family_peer"] == "test_held_out_family"
    # The ordinary family is pulled into the holdout closure because one semantic
    # cluster contains both ordinary and held-family members.
    assert result.assignments["train_rep"] == "test_held_out_family"
    assert set(result.splits["test_held_out_family"]) == {
        "family_member",
        "family_peer",
        "train_rep",
    }
    materialized = {record_id for split_ids in result.splits.values() for record_id in split_ids}
    assert "cross_source_exact" not in materialized
    assert "train_exact" not in materialized

    constraints = result.report["constraints"]
    assert constraints["no_held_out_source_in_train"] is True
    assert constraints["no_held_out_family_in_train"] is True
    assert constraints["no_template_family_overlap_with_train"] is True
    assert constraints["no_cluster_leakage"] is True
    assert constraints["exact_duplicates_excluded"] is True
    assert constraints["all_satisfied"] is True
    assert result.report["dropped_exact_ids"] == [
        "cross_source_exact",
        "train_exact",
    ]


def test_notinject_uses_test_favoring_ratio_policy() -> None:
    records = [
        _record("general", "promptinject"),
        _record("hard_negative", "notinject"),
    ]
    dedup = {
        "general": _dedup("cluster_general"),
        "hard_negative": _dedup("cluster_notinject"),
    }
    families = {
        "general": _family("override_previous_instructions"),
        "hard_negative": _family("quoted_attack_hard_negative"),
    }
    config = SplitConfig(
        seed=3,
        general_ratios={"train": 1.0, "validation": 0.0, "test_id": 0.0},
        notinject_ratios={"train": 0.0, "validation": 0.0, "test_id": 1.0},
    )

    result = assign_splits(records, dedup, families, config)

    assert result.assignments == {
        "general": "train",
        "hard_negative": "test_id",
    }
    policies = result.report["policy"]["ratio_policy_by_cluster"]
    assert policies == {
        "cluster_general": "general",
        "cluster_notinject": "notinject",
    }


def test_agentic_sources_and_cross_source_clusters_are_provisional_and_atomic() -> None:
    records = [
        _agentic_record(
            "injec-base",
            "injecagent",
            parent_group_id="injecagent:direct_harm:attacker-1:user-1",
        ),
        _agentic_record(
            "injec-enhanced",
            "injecagent",
            parent_group_id="injecagent:direct_harm:attacker-1:user-1",
        ),
        _agentic_record(
            "dojo-case",
            "agentdojo",
            parent_group_id="agentdojo:v1.2.2:workspace:user_task_0:injection_task_0",
        ),
        _record("legacy-near-duplicate", "promptinject"),
        _record("ordinary", "promptinject"),
    ]
    dedup = {
        "injec-base": _dedup("cluster-injec-parent"),
        "injec-enhanced": _dedup("cluster-injec-parent"),
        "dojo-case": _dedup("cluster-cross-source"),
        "legacy-near-duplicate": _dedup("cluster-cross-source"),
        "ordinary": _dedup("cluster-ordinary"),
    }
    families = {
        "injec-base": _family("injecagent_direct_harm_attacker_1"),
        "injec-enhanced": _family("injecagent_direct_harm_attacker_1"),
        "dojo-case": _family("agentdojo_workspace_injection_task_0"),
        "legacy-near-duplicate": _family("override_previous_instructions"),
        "ordinary": _family("ordinary"),
    }
    config = SplitConfig(
        seed=33,
        agentic_sources=frozenset({"injecagent", "agentdojo"}),
        general_ratios={"train": 1.0, "validation": 0.0, "test_id": 0.0},
        notinject_ratios={"train": 0.0, "validation": 0.0, "test_id": 1.0},
    )

    result = assign_splits(records, dedup, families, config)

    assert set(result.splits["test_agentic_provisional"]) == {
        "dojo-case",
        "injec-base",
        "injec-enhanced",
        "legacy-near-duplicate",
    }
    assert result.assignments["ordinary"] == "train"
    constraints = result.report["constraints"]
    assert constraints["no_cluster_leakage"] is True
    assert constraints["no_agentic_source_outside_provisional"] is True
    assert constraints["no_agentic_parent_group_leakage"] is True
    assert constraints["all_satisfied"] is True


def test_hash_assignment_is_deterministic_and_input_order_independent() -> None:
    records = [_record(f"record-{index:02d}", "promptinject") for index in range(30)]
    dedup = {record["id"]: _dedup(f"cluster-{record['id']}") for record in records}
    families = {record["id"]: _family("override_previous_instructions") for record in records}
    config = SplitConfig(
        seed=20260711,
        general_ratios={"train": 0.6, "validation": 0.2, "test_id": 0.2},
        notinject_ratios={"train": 0.2, "validation": 0.1, "test_id": 0.7},
    )

    forward = assign_splits(records, dedup, families, config)
    reversed_result = assign_splits(list(reversed(records)), dedup, families, config)

    assert forward.assignments == reversed_result.assignments
    assert forward.splits == reversed_result.splits
    assert forward.report == reversed_result.report
    assert sum(len(ids) for ids in forward.splits.values()) == 30
    assert set(forward.assignments.values()) <= {"train", "validation", "test_id"}


def test_dropped_exact_member_does_not_perturb_cluster_hash_assignment() -> None:
    representative = _record("representative", "promptinject")
    duplicate = _record("z-duplicate", "promptinject")
    families = {
        "representative": _family("override_previous_instructions"),
        "z-duplicate": _family("override_previous_instructions"),
    }
    config = SplitConfig(
        seed=41,
        general_ratios={"train": 0.5, "validation": 0.25, "test_id": 0.25},
        notinject_ratios={"train": 0.2, "validation": 0.1, "test_id": 0.7},
    )
    without_duplicate = assign_splits(
        [representative],
        {"representative": _dedup("stable_cluster")},
        {"representative": families["representative"]},
        config,
    )
    with_duplicate = assign_splits(
        [representative, duplicate],
        {
            "representative": _dedup("stable_cluster"),
            "z-duplicate": _dedup("stable_cluster", "DROP_EXACT_DUPLICATE"),
        },
        families,
        config,
    )

    assert (
        without_duplicate.assignments["representative"]
        == with_duplicate.assignments["representative"]
    )
    assert with_duplicate.assignments["z-duplicate"] == "DROPPED_EXACT"


def test_rejects_missing_assignments_and_invalid_ratios() -> None:
    record = _record("record", "promptinject")
    family = {"record": _family("override_previous_instructions")}

    with pytest.raises(SplitError, match="dedup_assignments missing record ids"):
        assign_splits([record], {}, family, SplitConfig())

    with pytest.raises(SplitError, match="must sum to 1.0"):
        SplitConfig(general_ratios={"train": 0.8, "validation": 0.2, "test_id": 0.2})
