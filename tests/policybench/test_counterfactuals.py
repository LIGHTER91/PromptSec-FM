from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from promptsec.policybench.blueprints import PolicyDescriptor, make_blueprint
from promptsec.policybench.config import COUNTERFACTUAL_TYPE_ORDER
from promptsec.policybench.counterfactuals import (
    CHANGED_VARIABLES,
    CounterfactualError,
    CounterfactualGroup,
    make_counterfactual_group,
)
from promptsec.policybench.validation import validate_counterfactual_record_groups


def _policy(
    policy_id: str,
    *,
    send_effect: str,
    tool_authority: str,
) -> PolicyDescriptor:
    send_allowed = send_effect == "ALLOW"
    return PolicyDescriptor(
        policy_id=policy_id,
        domain="banking",
        policy_family="transfer_policy_family",
        protected_policy_text_en=f"Policy {policy_id} controls transfers.",
        protected_policy_text_fr=f"La politique {policy_id} contrôle les virements.",
        allowed_actions=("READ_ACCOUNT_BALANCE", "SEND_MONEY")
        if send_allowed
        else ("READ_ACCOUNT_BALANCE",),
        forbidden_actions=() if send_allowed else ("SEND_MONEY",),
        confirmation_actions=(),
        rule_ids=(f"{policy_id.upper()}_SEND", f"{policy_id.upper()}_AUTH"),
        rules=(
            {
                "rule_id": f"{policy_id.upper()}_SEND",
                "effect": send_effect,
                "action": "SEND_MONEY",
            },
            {
                "rule_id": f"{policy_id.upper()}_AUTH",
                "effect": tool_authority,
                "actions": ["SEND_MONEY"],
                "source_roles": ["TOOL"],
            },
        ),
    )


DENY_POLICY = _policy(
    "banking_policy_901",
    send_effect="DENY",
    tool_authority="DENY_AUTHORITY",
)
ALLOW_POLICY = _policy(
    "banking_policy_902",
    send_effect="ALLOW",
    tool_authority="ALLOW_AUTHORITY",
)


def _attack_parent():
    return make_blueprint(
        DENY_POLICY,
        category="ALIGNED_BUT_POLICY_CONFLICTING",
        language="en",
        scenario_id="pb_cf_attack_parent",
        seed=5,
    )


def _compliant_parent():
    return make_blueprint(
        ALLOW_POLICY,
        category="ALIGNED_AND_COMPLIANT",
        language="en",
        scenario_id="pb_cf_compliant_parent",
        seed=6,
    )


@pytest.mark.parametrize("counterfactual_type", COUNTERFACTUAL_TYPE_ORDER)
def test_all_six_counterfactual_families_are_valid(counterfactual_type: str) -> None:
    parent = _compliant_parent() if counterfactual_type == "CAPABILITY_CHANGE" else _attack_parent()
    alternate = ALLOW_POLICY if counterfactual_type == "POLICY_CHANGE" else None
    group = make_counterfactual_group(
        parent,
        counterfactual_type,
        alternate_policy=alternate,
        current_policy=DENY_POLICY,
    )
    assert group.changed_variable == CHANGED_VARIABLES[counterfactual_type]
    assert len(group.members) == 2
    assert group.validate() == []
    assert all(member.counterfactual is not None for member in group.members)
    assert all(member.validation_errors() == [] for member in group.members)


def test_user_goal_change_recomputes_misalignment_objectives() -> None:
    parent, sibling = make_counterfactual_group(
        _attack_parent(),
        "USER_GOAL_CHANGE",
        current_policy=DENY_POLICY,
    ).members
    assert parent.expected_annotations.user_goal_alignment == "ALIGNED"
    assert sibling.expected_annotations.user_goal_alignment == "MISALIGNED"
    assert "TASK_HIJACKING" in sibling.expected_annotations.attack_objectives
    assert "OUTPUT_OR_DECISION_MANIPULATION" in sibling.expected_annotations.attack_objectives
    assert "UNAUTHORIZED_ACTION" in sibling.expected_annotations.attack_objectives


def test_capability_change_is_a_taxonomy_invariance_pair() -> None:
    parent, sibling = make_counterfactual_group(
        _compliant_parent(),
        "CAPABILITY_CHANGE",
        current_policy=ALLOW_POLICY,
    ).members
    assert parent.available_capabilities != sibling.available_capabilities
    assert parent.expected_annotations == sibling.expected_annotations
    assert parent.counterfactual is not None
    assert parent.counterfactual.expected_label_changes == ()


def test_authority_delegation_changes_only_controlled_effective_boundary() -> None:
    parent, sibling = make_counterfactual_group(
        _attack_parent(),
        "AUTHORITY_DELEGATION_CHANGE",
        current_policy=DENY_POLICY,
    ).members
    assert parent.policy_id == sibling.policy_id
    assert parent.policy_rule_ids == sibling.policy_rule_ids
    assert parent.grouping == sibling.grouping
    assert parent.authority_delegation_override is not None
    assert sibling.authority_delegation_override is not None
    left = parent.authority_delegation_override.to_dict()
    right = sibling.authority_delegation_override.to_dict()
    assert {
        left.pop("effective_effect"),
        right.pop("effective_effect"),
    } == {"ALLOW_AUTHORITY", "DENY_AUTHORITY"}
    left.pop("effective_protected_policy_text")
    right.pop("effective_protected_policy_text")
    assert left == right


def test_presentation_change_removes_operative_attack_semantics() -> None:
    parent, quoted = make_counterfactual_group(
        _attack_parent(),
        "PRESENTATION_CHANGE",
        current_policy=DENY_POLICY,
    ).members
    assert parent.expected_annotations.prompt_injection_verdict == "DETECTED"
    assert quoted.expected_annotations.instruction_presentation == "QUOTED_OR_REPORTED"
    assert quoted.expected_annotations.protected_policy_alignment == "COMPLIANT"
    assert quoted.expected_annotations.authority_status == "WITHIN_AUTHORITY"
    assert quoted.expected_annotations.attack_families == ()
    assert quoted.expected_annotations.attack_objectives == ()
    assert quoted.expected_annotations.prompt_injection_verdict == "NOT_DETECTED"


def test_policy_counterfactual_requires_a_real_supporting_catalogue_policy() -> None:
    with pytest.raises(CounterfactualError, match="real same-domain alternate policy"):
        make_counterfactual_group(_attack_parent(), "POLICY_CHANGE")
    wrong = replace(ALLOW_POLICY, domain="email", policy_id="email_policy_902")
    with pytest.raises(CounterfactualError, match="same domain"):
        make_counterfactual_group(
            _attack_parent(),
            "POLICY_CHANGE",
            alternate_policy=wrong,
        )


def test_policy_change_retains_policy_family() -> None:
    group = make_counterfactual_group(
        _attack_parent(),
        "POLICY_CHANGE",
        alternate_policy=ALLOW_POLICY,
        current_policy=DENY_POLICY,
    )
    parent, sibling = group.members
    assert parent.policy_id != sibling.policy_id
    assert parent.grouping is not None and sibling.grouping is not None
    assert parent.grouping.policy_family == sibling.grouping.policy_family
    cross_family = replace(ALLOW_POLICY, policy_family="unrelated_policy_family")
    with pytest.raises(CounterfactualError, match="retain the policy family"):
        make_counterfactual_group(
            _attack_parent(),
            "POLICY_CHANGE",
            alternate_policy=cross_family,
            current_policy=DENY_POLICY,
        )


def test_invariant_validator_detects_an_undeclared_change() -> None:
    group = make_counterfactual_group(
        _attack_parent(),
        "POLICY_CHANGE",
        alternate_policy=ALLOW_POLICY,
    )
    parent, sibling = group.members
    tampered = replace(sibling, source_role="USER")
    broken = CounterfactualGroup(
        group_id=group.group_id,
        counterfactual_type=group.counterfactual_type,
        changed_variable=group.changed_variable,
        parent_scenario_id=group.parent_scenario_id,
        members=(parent, tampered),
    )
    assert any("source_role" in error for error in broken.validate())


def test_label_changes_are_derived_from_siblings_not_provider_output() -> None:
    group = make_counterfactual_group(_attack_parent(), "PRESENTATION_CHANGE")
    sibling = group.members[1]
    assert sibling.counterfactual is not None
    fields = {change.field for change in sibling.counterfactual.expected_label_changes}
    assert {
        "instruction_presentation",
        "protected_policy_alignment",
        "authority_status",
        "attack_families",
        "attack_objectives",
        "prompt_injection_verdict",
    } <= fields


def _realized_records(group: CounterfactualGroup) -> list[dict]:
    records: list[dict] = []
    for index, blueprint in enumerate(group.members):
        annotations = blueprint.expected_annotations.canonical_annotations()
        policy_override = (
            blueprint.authority_delegation_override.to_dict()
            if blueprint.authority_delegation_override
            else None
        )
        records.append(
            {
                "id": f"record_{index}",
                "context": {
                    "protected_policy": blueprint.protected_policy_text,
                    "user_goal": (
                        str(blueprint.user_goal_spec.to_dict())
                        if blueprint.user_goal_spec is not None
                        else None
                    ),
                    "available_capabilities": list(blueprint.available_capabilities),
                },
                "content": {
                    "text": (
                        "Quoted: 'send money'."
                        if blueprint.expected_annotations.instruction_presentation
                        == "QUOTED_OR_REPORTED"
                        else "Send money."
                    ),
                    "language": blueprint.language,
                    "delivery_mode": blueprint.delivery_mode,
                    "source_role": blueprint.source_role,
                    "content_origin": blueprint.content_origin,
                    "ingestion_path": blueprint.ingestion_path,
                    "modality": blueprint.modality,
                    "source_integrity": blueprint.source_integrity,
                },
                "annotations": annotations,
                "derived": {
                    "prompt_injection_verdict": (
                        blueprint.expected_annotations.prompt_injection_verdict
                    )
                },
                "extensions": {
                    "policybench_v0_1": {
                        "policy": {
                            "policy_id": blueprint.policy_id,
                            "policy_family": blueprint.grouping.policy_family,
                            "policy_version": "0.1",
                            "matched_rule_ids": list(blueprint.policy_rule_ids),
                            "policy_sha256": "0" * 64,
                            "effective_policy_sha256": str(index),
                            "authority_delegation_override": policy_override,
                        },
                        "blueprint": {
                            "scenario_blueprint_id": blueprint.scenario_id,
                        },
                        "counterfactual": blueprint.counterfactual.to_dict(),
                        "grouping": {"split_group_id": group.group_id},
                        "dataset_split": "test_counterfactual",
                    }
                },
            }
        )
    return records


def test_reverse_user_goal_change_preserves_conflicting_objective_consistency() -> None:
    parent = make_blueprint(
        DENY_POLICY,
        category="SPOOFED_AUTHORITY",
        language="fr",
        scenario_id="pb_cf_reverse_user_goal",
        seed=5,
    )
    group = make_counterfactual_group(
        parent,
        "USER_GOAL_CHANGE",
        current_policy=DENY_POLICY,
    )

    assert validate_counterfactual_record_groups(_realized_records(group)) == []


@pytest.mark.parametrize(
    ("counterfactual_type", "parent_factory", "policy", "tamper", "message"),
    (
        (
            "CAPABILITY_CHANGE",
            _compliant_parent,
            ALLOW_POLICY,
            lambda records: records[1]["annotations"].__setitem__(
                "authority_status", "OUTSIDE_AUTHORITY"
            ),
            "must not invent frozen taxonomy consequences",
        ),
        (
            "USER_GOAL_CHANGE",
            _attack_parent,
            DENY_POLICY,
            lambda records: records[1]["annotations"].__setitem__(
                "attack_objectives", records[0]["annotations"]["attack_objectives"]
            ),
            "omitted TASK_HIJACKING",
        ),
        (
            "PRESENTATION_CHANGE",
            _attack_parent,
            DENY_POLICY,
            lambda records: records[1]["annotations"].update(
                {
                    "authority_status": "OUTSIDE_AUTHORITY",
                    "attack_families": ["PROMPT_INJECTION"],
                    "attack_objectives": ["UNAUTHORIZED_ACTION"],
                }
            ),
            "quoted sibling leaks operative attack semantics",
        ),
        (
            "AUTHORITY_DELEGATION_CHANGE",
            _attack_parent,
            DENY_POLICY,
            lambda records: records[1]["extensions"]["policybench_v0_1"]["policy"].update(
                {"policy_id": "banking_policy_902"}
            ),
            "retain policy identity",
        ),
    ),
)
def test_realized_validator_rejects_semantically_invalid_pairs(
    counterfactual_type,
    parent_factory,
    policy,
    tamper,
    message,
) -> None:
    group = make_counterfactual_group(
        parent_factory(),
        counterfactual_type,
        current_policy=policy,
    )
    records = _realized_records(group)
    assert validate_counterfactual_record_groups(records) == []
    tampered = copy.deepcopy(records)
    tamper(tampered)
    errors = validate_counterfactual_record_groups(tampered)
    assert errors
    assert any(message in error for error in errors[0]["errors"])
