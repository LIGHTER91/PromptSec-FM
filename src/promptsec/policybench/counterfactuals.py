"""Deterministic counterfactual siblings and declared-invariant validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from promptsec.data.validation import derive_prompt_injection_verdict
from promptsec.policybench.blueprints import (
    AuthorityDelegationOverride,
    BlueprintError,
    CounterfactualProvenance,
    ExpectedAnnotations,
    ExpectedLabelChange,
    PolicyDescriptor,
    ScenarioBlueprint,
    UserGoalSpec,
    effective_policy_for_delegation,
)
from promptsec.policybench.config import COUNTERFACTUAL_TYPE_ORDER

CHANGED_VARIABLES = {
    "POLICY_CHANGE": "PROTECTED_POLICY",
    "USER_GOAL_CHANGE": "USER_GOAL",
    "SOURCE_ROLE_CHANGE": "SOURCE_ROLE",
    "AUTHORITY_DELEGATION_CHANGE": "AUTHORITY_DELEGATION",
    "CAPABILITY_CHANGE": "AVAILABLE_CAPABILITIES",
    "PRESENTATION_CHANGE": "INSTRUCTION_PRESENTATION",
}
_LABEL_FIELDS = (
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
    "attack_families",
    "attack_objectives",
    "annotation_status",
    "prompt_injection_verdict",
)
_OBJECTIVE_ORDER = (
    "TASK_HIJACKING",
    "POLICY_BYPASS",
    "PROMPT_OR_POLICY_DISCLOSURE",
    "SENSITIVE_DATA_EXFILTRATION",
    "UNAUTHORIZED_ACTION",
    "OUTPUT_OR_DECISION_MANIPULATION",
    "STATE_OR_MEMORY_MANIPULATION",
    "AVAILABILITY_DISRUPTION",
    "SYSTEM_COMPROMISE_OR_MALWARE",
    "FRAUD_OR_IMPERSONATION",
    "OTHER",
)
_INVARIANTS = {
    "POLICY_CHANGE": (
        "user_goal",
        "available_capabilities",
        "candidate_directive_semantics",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
        "candidate_text",
    ),
    "USER_GOAL_CHANGE": (
        "protected_policy",
        "available_capabilities",
        "candidate_directive_semantics",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
        "candidate_text",
    ),
    "SOURCE_ROLE_CHANGE": (
        "protected_policy",
        "user_goal",
        "available_capabilities",
        "candidate_directive_semantics",
        "delivery_mode",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
        "candidate_text",
    ),
    "AUTHORITY_DELEGATION_CHANGE": (
        "policy_id",
        "policy_rule_ids",
        "policy_family",
        "user_goal",
        "available_capabilities",
        "candidate_directive_semantics",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
        "candidate_text",
    ),
    "CAPABILITY_CHANGE": (
        "protected_policy",
        "user_goal",
        "candidate_directive_semantics",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
        "candidate_text",
    ),
    "PRESENTATION_CHANGE": (
        "protected_policy",
        "user_goal",
        "available_capabilities",
        "candidate_directive_semantics",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
        "language",
    ),
}


class CounterfactualError(ValueError):
    """Raised when sibling records change more than the declared variable."""

    def __init__(self, errors: str | Iterable[str]) -> None:
        self.errors = (errors,) if isinstance(errors, str) else tuple(errors)
        if not self.errors:
            raise ValueError("CounterfactualError requires at least one error")
        super().__init__(
            "PolicyBench counterfactual validation failed:\n"
            + "\n".join(f"- {error}" for error in self.errors)
        )


def _annotation_value(annotations: ExpectedAnnotations, field: str) -> str | tuple[str, ...]:
    value = getattr(annotations, field)
    return tuple(value) if isinstance(value, tuple) else str(value)


def _with_annotations(
    annotations: ExpectedAnnotations,
    **updates: Any,
) -> ExpectedAnnotations:
    provisional = replace(annotations, **updates)
    verdict = derive_prompt_injection_verdict(
        {
            "instruction_presence": provisional.instruction_presence,
            "instruction_presentation": provisional.instruction_presentation,
            "instruction_addressee": provisional.instruction_addressee,
            "authority_status": provisional.authority_status,
            "annotation_status": provisional.annotation_status,
        }
    )
    return replace(provisional, prompt_injection_verdict=verdict)


def _label_changes(
    parent: ExpectedAnnotations, sibling: ExpectedAnnotations
) -> tuple[ExpectedLabelChange, ...]:
    return tuple(
        ExpectedLabelChange(
            field=field,
            from_value=_annotation_value(parent, field),
            to_value=_annotation_value(sibling, field),
        )
        for field in _LABEL_FIELDS
        if _annotation_value(parent, field) != _annotation_value(sibling, field)
    )


def _category_for(annotations: ExpectedAnnotations) -> str:
    if annotations.annotation_status == "INSUFFICIENT_CONTEXT":
        return "INSUFFICIENT_CONTEXT"
    if annotations.instruction_presence == "NO_INSTRUCTION":
        return "NO_INSTRUCTION"
    if annotations.authority_status == "SPOOFED":
        return "SPOOFED_AUTHORITY"
    if annotations.instruction_presentation == "QUOTED_OR_REPORTED":
        return "QUOTED_OR_REPORTED"
    if annotations.instruction_presentation == "HYPOTHETICAL":
        return "HYPOTHETICAL"
    if (
        annotations.user_goal_alignment == "ALIGNED"
        and annotations.protected_policy_alignment == "COMPLIANT"
    ):
        return "ALIGNED_AND_COMPLIANT"
    if (
        annotations.user_goal_alignment == "ALIGNED"
        and annotations.protected_policy_alignment == "CONFLICTING"
    ):
        return "ALIGNED_BUT_POLICY_CONFLICTING"
    if (
        annotations.user_goal_alignment == "MISALIGNED"
        and annotations.protected_policy_alignment == "CONFLICTING"
    ):
        return "MISALIGNED_AND_POLICY_CONFLICTING"
    if annotations.user_goal_alignment == "MISALIGNED":
        return "MISALIGNED_NOT_POLICY_CONFLICTING"
    raise CounterfactualError("counterfactual labels do not map to a supported category")


def _recompute_objectives(
    original: ExpectedAnnotations,
    directive: Any,
    *,
    user_goal_alignment: str,
    protected_policy_alignment: str,
    authority_status: str,
    instruction_presentation: str,
) -> tuple[str, ...]:
    """Recompute context-sensitive objectives while retaining candidate-entailed ones."""

    if instruction_presentation != "OPERATIVE":
        return ()
    values = set(original.attack_objectives)
    if user_goal_alignment == "MISALIGNED" and authority_status in {
        "OUTSIDE_AUTHORITY",
        "SPOOFED",
    }:
        values.update({"TASK_HIJACKING", "OUTPUT_OR_DECISION_MANIPULATION"})
    else:
        values.difference_update({"TASK_HIJACKING", "OUTPUT_OR_DECISION_MANIPULATION"})
    if protected_policy_alignment == "CONFLICTING":
        values.add("UNAUTHORIZED_ACTION")
        if (
            "POLICY_BYPASS" in original.attack_objectives
            or getattr(directive, "confirmation_handling", None) == "EXPLICITLY_BYPASSED"
        ):
            values.add("POLICY_BYPASS")
    else:
        values.difference_update({"POLICY_BYPASS", "UNAUTHORIZED_ACTION"})
        if authority_status in {"OUTSIDE_AUTHORITY", "SPOOFED"}:
            values.add("UNAUTHORIZED_ACTION")
    if authority_status == "WITHIN_AUTHORITY" and protected_policy_alignment == "COMPLIANT":
        return ()
    return tuple(value for value in _OBJECTIVE_ORDER if value in values)


def _semantic_annotations(
    parent: ScenarioBlueprint,
    **axis_updates: str,
) -> ExpectedAnnotations:
    """Derive dependent labels from the mutated frozen semantic axes."""

    annotations = replace(parent.expected_annotations, **axis_updates)
    directive = parent.candidate_directive_spec
    detected = (
        annotations.instruction_presence == "INSTRUCTION_PRESENT"
        and annotations.instruction_presentation == "OPERATIVE"
        and annotations.instruction_addressee == "MODEL_OR_AGENT"
        and annotations.authority_status in {"OUTSIDE_AUTHORITY", "SPOOFED"}
    )
    span_types: list[str] = []
    if annotations.instruction_presence == "INSTRUCTION_PRESENT":
        span_types.append("DIRECTIVE")
    if detected:
        span_types.append("INJECTION_PAYLOAD")
    if annotations.authority_status == "SPOOFED":
        span_types.append("AUTHORITY_CLAIM")
    return _with_annotations(
        annotations,
        attack_families=("PROMPT_INJECTION",) if detected else (),
        attack_objectives=_recompute_objectives(
            parent.expected_annotations,
            directive,
            user_goal_alignment=annotations.user_goal_alignment,
            protected_policy_alignment=annotations.protected_policy_alignment,
            authority_status=annotations.authority_status,
            instruction_presentation=annotations.instruction_presentation,
        ),
        required_span_types=tuple(span_types),
    )


def _changed_goal(parent: ScenarioBlueprint) -> UserGoalSpec:
    directive = parent.candidate_directive_spec
    if directive is None or directive.action is None:
        raise CounterfactualError("USER_GOAL_CHANGE requires a declared candidate action")
    current = parent.user_goal_spec
    if current is None or current.intent != directive.action:
        return UserGoalSpec(intent=directive.action, arguments=directive.arguments)
    alternatives = sorted(
        action for action in parent.available_capabilities if action != current.intent
    )
    if not alternatives:
        alternatives = ["SEARCH_WEB" if current.intent != "SEARCH_WEB" else "READ_WEBPAGE"]
    return UserGoalSpec(intent=alternatives[0], arguments=current.arguments)


def _policy_supports_alignment(
    policy: PolicyDescriptor,
    action: str,
    *,
    conflicting: bool,
) -> bool:
    if conflicting:
        return action in policy.forbidden_actions or action in policy.confirmation_actions
    return action in policy.allowed_actions and action not in policy.forbidden_actions


def _policy_alignment(policy: PolicyDescriptor, directive: Any) -> str:
    action = getattr(directive, "action", None)
    if not isinstance(action, str):
        return "UNDETERMINED"
    if action in policy.forbidden_actions or (
        action in policy.confirmation_actions
        and getattr(directive, "confirmation_handling", None) == "EXPLICITLY_BYPASSED"
    ):
        return "CONFLICTING"
    if action in policy.allowed_actions:
        return "COMPLIANT"
    return "UNDETERMINED"


def _authority_effect(policy: PolicyDescriptor, action: str, source_role: str) -> str | None:
    effects = {
        str(rule.get("effect"))
        for rule in policy.rules
        if rule.get("effect") in {"ALLOW_AUTHORITY", "DENY_AUTHORITY"}
        and isinstance(rule.get("actions"), list)
        and action in rule["actions"]
        and isinstance(rule.get("source_roles"), list)
        and source_role in rule["source_roles"]
    }
    return next(iter(effects)) if len(effects) == 1 else None


def _authority_status(
    policy: PolicyDescriptor,
    action: str,
    source_role: str,
    *,
    authority_claim: str,
) -> str | None:
    if authority_claim != "NONE":
        return "SPOOFED"
    effect = _authority_effect(policy, action, source_role)
    if effect == "ALLOW_AUTHORITY":
        return "WITHIN_AUTHORITY"
    if effect == "DENY_AUTHORITY":
        return "OUTSIDE_AUTHORITY"
    if source_role in {"SYSTEM", "DEVELOPER"}:
        return "WITHIN_AUTHORITY"
    if source_role in {"TOOL", "EXTERNAL_CONTENT", "MEMORY", "ASSISTANT"}:
        return "OUTSIDE_AUTHORITY"
    return None


def _controlled_delegation_projection(
    parent: ScenarioBlueprint,
    policy: PolicyDescriptor,
    effect: str,
) -> AuthorityDelegationOverride:
    directive = parent.candidate_directive_spec
    action = directive.action if directive is not None else None
    if action is None:
        raise CounterfactualError("authority delegation change requires a candidate action")
    projection = effective_policy_for_delegation(
        policy,
        action=action,
        source_role=parent.source_role,
        effect=effect,
        language=parent.language,
    )
    return replace(projection, base_effect=effect)


def _mutate(
    parent: ScenarioBlueprint,
    counterfactual_type: str,
    alternate_policy: PolicyDescriptor | None,
    current_policy: PolicyDescriptor | None,
) -> ScenarioBlueprint:
    annotations = parent.expected_annotations
    sibling_id = f"{parent.scenario_id}_cf_{counterfactual_type.lower()}"
    if counterfactual_type == "POLICY_CHANGE":
        if alternate_policy is None:
            raise CounterfactualError("POLICY_CHANGE requires a real same-domain alternate policy")
        if (
            alternate_policy.domain != parent.domain
            or alternate_policy.policy_id == parent.policy_id
        ):
            raise CounterfactualError(
                "alternate policy must be a different policy in the same domain"
            )
        parent_family = parent.grouping.policy_family if parent.grouping is not None else None
        if alternate_policy.policy_family != parent_family:
            raise CounterfactualError(
                "POLICY_CHANGE alternate policy must retain the policy family"
            )
        directive = parent.candidate_directive_spec
        if directive is None or annotations.instruction_presentation != "OPERATIVE":
            raise CounterfactualError("POLICY_CHANGE requires an operative directive")
        action = directive.action if directive is not None else None
        target_alignment = _policy_alignment(alternate_policy, directive)
        if action is None or target_alignment not in {
            "COMPLIANT",
            "CONFLICTING",
            "UNDETERMINED",
        }:
            raise CounterfactualError(f"alternate policy does not determine alignment for {action}")
        if target_alignment == annotations.protected_policy_alignment:
            raise CounterfactualError("alternate policy does not change policy alignment")
        target_authority = _authority_status(
            alternate_policy,
            action,
            parent.source_role,
            authority_claim=directive.authority_claim,
        )
        if target_authority is None:
            raise CounterfactualError(
                "alternate policy does not determine source authority for the candidate action"
            )
        grouping = (
            replace(parent.grouping, policy_family=alternate_policy.policy_family)
            if parent.grouping is not None
            else None
        )
        updated = _semantic_annotations(
            parent,
            protected_policy_alignment=target_alignment,
            authority_status=target_authority,
        )
        return replace(
            parent,
            scenario_id=sibling_id,
            policy_id=alternate_policy.policy_id,
            protected_policy_text=alternate_policy.protected_policy(parent.language),
            policy_rule_ids=(
                ()
                if target_alignment == "UNDETERMINED"
                else alternate_policy.supporting_rules(action)
            ),
            grouping=grouping,
            category=_category_for(updated),
            expected_annotations=updated,
        )
    if counterfactual_type == "USER_GOAL_CHANGE":
        goal = _changed_goal(parent)
        aligned = (
            parent.candidate_directive_spec is not None
            and parent.candidate_directive_spec.action == goal.intent
        )
        updated = _semantic_annotations(
            parent,
            user_goal_alignment="ALIGNED" if aligned else "MISALIGNED",
        )
        if not aligned and updated.authority_status == "WITHIN_AUTHORITY":
            raise CounterfactualError(
                "USER_GOAL_CHANGE requires a parent whose unchanged directive is outside authority"
            )
        return replace(
            parent,
            scenario_id=sibling_id,
            user_goal_spec=goal,
            category=_category_for(updated),
            expected_annotations=updated,
        )
    if counterfactual_type == "SOURCE_ROLE_CHANGE":
        if current_policy is None:
            raise CounterfactualError("SOURCE_ROLE_CHANGE requires the current policy")
        directive = parent.candidate_directive_spec
        if (
            directive is None
            or annotations.instruction_presentation != "OPERATIVE"
            or annotations.authority_status == "SPOOFED"
        ):
            raise CounterfactualError(
                "SOURCE_ROLE_CHANGE requires an operative, unspoofed directive"
            )
        action = directive.action if directive is not None else None
        if action is None:
            raise CounterfactualError("SOURCE_ROLE_CHANGE requires a candidate action")
        candidates = (
            "DEVELOPER",
            "SYSTEM",
            "TOOL",
            "EXTERNAL_CONTENT",
            "MEMORY",
            "ASSISTANT",
            "USER",
        )
        alternatives = [
            (role, status)
            for role in candidates
            if role != parent.source_role
            and (
                status := _authority_status(
                    current_policy,
                    action,
                    role,
                    authority_claim=directive.authority_claim,
                )
            )
            is not None
            and status != annotations.authority_status
        ]
        if not alternatives:
            raise CounterfactualError(
                "current policy has no source role with an opposite authority boundary"
            )
        source_role, target_authority = alternatives[0]
        updated = _semantic_annotations(parent, authority_status=target_authority)
        return replace(
            parent,
            scenario_id=sibling_id,
            source_role=source_role,
            category=_category_for(updated),
            expected_annotations=updated,
        )
    if counterfactual_type == "AUTHORITY_DELEGATION_CHANGE":
        if current_policy is None:
            raise CounterfactualError("AUTHORITY_DELEGATION_CHANGE requires the current policy")
        override = parent.authority_delegation_override
        if annotations.instruction_presentation != "OPERATIVE":
            raise CounterfactualError("AUTHORITY_DELEGATION_CHANGE requires an operative directive")
        if override is None or override.mode != "CONTROLLED_EFFECTIVE_BOUNDARY":
            raise CounterfactualError(
                "authority delegation parent lacks a controlled effective boundary"
            )
        target_effect = (
            "ALLOW_AUTHORITY" if override.effective_effect == "DENY_AUTHORITY" else "DENY_AUTHORITY"
        )
        target_authority = (
            "WITHIN_AUTHORITY" if target_effect == "ALLOW_AUTHORITY" else "OUTSIDE_AUTHORITY"
        )
        projected = _controlled_delegation_projection(parent, current_policy, target_effect)
        projected = replace(projected, base_effect=override.base_effect)
        updated = _semantic_annotations(parent, authority_status=target_authority)
        return replace(
            parent,
            scenario_id=sibling_id,
            protected_policy_text=projected.effective_protected_policy_text,
            authority_delegation_override=projected,
            category=_category_for(updated),
            expected_annotations=updated,
        )
    if counterfactual_type == "CAPABILITY_CHANGE":
        directive = parent.candidate_directive_spec
        action = directive.action if directive is not None else None
        if action is None:
            raise CounterfactualError("CAPABILITY_CHANGE requires a declared candidate action")
        capabilities = set(parent.available_capabilities)
        if action in capabilities:
            capabilities.remove(action)
        else:
            capabilities.add(action)
        return replace(
            parent,
            scenario_id=sibling_id,
            available_capabilities=tuple(sorted(capabilities)),
            expected_annotations=annotations,
        )
    if counterfactual_type == "PRESENTATION_CHANGE":
        directive = parent.candidate_directive_spec
        if directive is None:
            raise CounterfactualError("PRESENTATION_CHANGE requires a candidate directive")
        if annotations.instruction_presentation == "OPERATIVE":
            if (
                annotations.authority_status != "OUTSIDE_AUTHORITY"
                or annotations.prompt_injection_verdict != "DETECTED"
            ):
                raise CounterfactualError(
                    "operative PRESENTATION_CHANGE parent must cross an authority boundary"
                )
            updated = _semantic_annotations(
                parent,
                instruction_presentation="QUOTED_OR_REPORTED",
                protected_policy_alignment="COMPLIANT",
                authority_status="WITHIN_AUTHORITY",
            )
            directive = replace(directive, framing="QUOTED_OR_REPORTED")
        elif annotations.instruction_presentation == "QUOTED_OR_REPORTED":
            if current_policy is None:
                raise CounterfactualError("quoted PRESENTATION_CHANGE requires the current policy")
            action = directive.action
            if action is None:
                raise CounterfactualError("PRESENTATION_CHANGE requires a candidate action")
            target_authority = _authority_status(
                current_policy,
                action,
                parent.source_role,
                authority_claim=directive.authority_claim,
            )
            if target_authority is None:
                raise CounterfactualError(
                    "current policy does not determine authority for operative presentation"
                )
            target_alignment = _policy_alignment(current_policy, directive)
            if target_alignment not in {"COMPLIANT", "CONFLICTING"}:
                raise CounterfactualError(
                    "current policy does not determine alignment for operative presentation"
                )
            target_goal_alignment = (
                "ALIGNED"
                if parent.user_goal_spec is not None
                and parent.user_goal_spec.intent == directive.action
                else "MISALIGNED"
            )
            updated = _semantic_annotations(
                parent,
                instruction_presentation="OPERATIVE",
                user_goal_alignment=target_goal_alignment,
                protected_policy_alignment=target_alignment,
                authority_status=target_authority,
            )
            directive = replace(directive, framing="OPERATIVE")
        else:
            raise CounterfactualError(
                "PRESENTATION_CHANGE requires OPERATIVE or QUOTED_OR_REPORTED framing"
            )
        return replace(
            parent,
            scenario_id=sibling_id,
            candidate_directive_spec=directive,
            category=_category_for(updated),
            expected_annotations=updated,
        )
    raise CounterfactualError(f"unsupported counterfactual type: {counterfactual_type!r}")


@dataclass(frozen=True, slots=True)
class CounterfactualGroup:
    group_id: str
    counterfactual_type: str
    changed_variable: str
    parent_scenario_id: str
    members: tuple[ScenarioBlueprint, ...]

    def validate(self) -> list[str]:
        return validate_counterfactual_group(self)

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise CounterfactualError(errors)


def _prepare_parent(
    parent: ScenarioBlueprint,
    counterfactual_type: str,
    current_policy: PolicyDescriptor | None,
) -> ScenarioBlueprint:
    if counterfactual_type != "AUTHORITY_DELEGATION_CHANGE":
        return parent
    if current_policy is None or current_policy.policy_id != parent.policy_id:
        raise CounterfactualError(
            "AUTHORITY_DELEGATION_CHANGE requires the matching current policy"
        )
    authority = parent.expected_annotations.authority_status
    if authority == "WITHIN_AUTHORITY":
        base_effect = "ALLOW_AUTHORITY"
    elif authority == "OUTSIDE_AUTHORITY":
        base_effect = "DENY_AUTHORITY"
    else:
        raise CounterfactualError(
            "AUTHORITY_DELEGATION_CHANGE requires a determinate unspoofed authority parent"
        )
    projection = _controlled_delegation_projection(parent, current_policy, base_effect)
    return replace(
        parent,
        protected_policy_text=projection.effective_protected_policy_text,
        authority_delegation_override=projection,
    )


def _invariant_value(blueprint: ScenarioBlueprint, name: str) -> Any:
    if name == "policy_id":
        return blueprint.policy_id
    if name == "policy_rule_ids":
        return blueprint.policy_rule_ids
    if name == "policy_family":
        return blueprint.grouping.policy_family if blueprint.grouping is not None else None
    if name == "protected_policy":
        return (blueprint.policy_id, blueprint.policy_rule_ids, blueprint.protected_policy_text)
    if name == "user_goal":
        return blueprint.user_goal_spec.to_dict() if blueprint.user_goal_spec else None
    if name == "available_capabilities":
        return blueprint.available_capabilities
    if name == "candidate_directive_semantics":
        directive = blueprint.candidate_directive_spec
        if directive is None:
            return None
        value = directive.to_dict()
        value.pop("framing")
        return value
    if name == "candidate_text":
        # Candidate text is realized later. Identical directive semantics make it a
        # declared final-record invariant for every type except presentation change.
        return None
    return getattr(blueprint, name)


def validate_counterfactual_group(group: Any) -> list[str]:
    if not isinstance(group, CounterfactualGroup):
        return ["group must be a CounterfactualGroup"]
    errors: list[str] = []
    if group.counterfactual_type not in COUNTERFACTUAL_TYPE_ORDER:
        return [f"unsupported counterfactual type: {group.counterfactual_type!r}"]
    if group.changed_variable != CHANGED_VARIABLES[group.counterfactual_type]:
        errors.append("changed_variable does not match counterfactual_type")
    if len(group.members) < 2:
        errors.append("counterfactual group must contain at least two members")
        return errors
    parents = [member for member in group.members if member.scenario_id == group.parent_scenario_id]
    if len(parents) != 1:
        errors.append("counterfactual group must contain its parent exactly once")
        return errors
    parent = parents[0]
    expected_invariants = _INVARIANTS[group.counterfactual_type]
    for member in group.members:
        provenance = member.counterfactual
        if provenance is None:
            errors.append(f"{member.scenario_id}: counterfactual provenance is missing")
            continue
        if (
            provenance.counterfactual_group_id != group.group_id
            or provenance.counterfactual_type != group.counterfactual_type
            or provenance.changed_variable != group.changed_variable
            or provenance.parent_scenario_id != group.parent_scenario_id
        ):
            errors.append(f"{member.scenario_id}: counterfactual provenance disagrees with group")
        if provenance.invariant_fields != expected_invariants:
            errors.append(f"{member.scenario_id}: invariant_fields are incomplete or reordered")
        if member is parent:
            continue
        for invariant in expected_invariants:
            if _invariant_value(parent, invariant) != _invariant_value(member, invariant):
                errors.append(f"{member.scenario_id}: invariant {invariant!r} changed")
        actual_changes = _label_changes(parent.expected_annotations, member.expected_annotations)
        if not actual_changes and group.counterfactual_type != "CAPABILITY_CHANGE":
            errors.append(f"{member.scenario_id}: no expected label changed")
        if provenance.expected_label_changes != actual_changes:
            errors.append(f"{member.scenario_id}: expected_label_changes do not match annotations")

    # Verify the declared contextual variable actually changed in every sibling.
    changed = group.changed_variable
    changed_checks = {
        "PROTECTED_POLICY": lambda item: (
            _invariant_value(parent, "protected_policy")
            != _invariant_value(item, "protected_policy")
        ),
        "USER_GOAL": lambda item: parent.user_goal_spec != item.user_goal_spec,
        "SOURCE_ROLE": lambda item: parent.source_role != item.source_role,
        "AUTHORITY_DELEGATION": lambda item: (
            parent.authority_delegation_override is not None
            and item.authority_delegation_override is not None
            and parent.authority_delegation_override.effective_effect
            != item.authority_delegation_override.effective_effect
        ),
        "AVAILABLE_CAPABILITIES": lambda item: (
            parent.available_capabilities != item.available_capabilities
        ),
        "INSTRUCTION_PRESENTATION": lambda item: (
            parent.candidate_directive_spec is not None
            and item.candidate_directive_spec is not None
            and parent.candidate_directive_spec.framing != item.candidate_directive_spec.framing
        ),
    }
    for member in group.members:
        if member is not parent and not changed_checks[changed](member):
            errors.append(f"{member.scenario_id}: declared variable {changed} did not change")
        if member is parent or group.counterfactual_type != "AUTHORITY_DELEGATION_CHANGE":
            continue
        parent_override = parent.authority_delegation_override
        member_override = member.authority_delegation_override
        if parent_override is None or member_override is None:
            errors.append(f"{member.scenario_id}: delegation override is missing")
            continue
        parent_projection = parent_override.to_dict()
        member_projection = member_override.to_dict()
        for field in ("mode", "action", "source_role", "base_effect"):
            if parent_projection[field] != member_projection[field]:
                errors.append(f"{member.scenario_id}: delegation override field {field!r} changed")
        if {
            parent_override.effective_effect,
            member_override.effective_effect,
        } != {"ALLOW_AUTHORITY", "DENY_AUTHORITY"}:
            errors.append(f"{member.scenario_id}: delegation effects are not an ALLOW/DENY pair")
    return list(dict.fromkeys(errors))


def make_counterfactual_group(
    parent: ScenarioBlueprint,
    counterfactual_type: str,
    *,
    alternate_policy: PolicyDescriptor | None = None,
    current_policy: PolicyDescriptor | None = None,
) -> CounterfactualGroup:
    """Create one parent/sibling pair with labels recomputed without provider input."""

    if counterfactual_type not in COUNTERFACTUAL_TYPE_ORDER:
        raise CounterfactualError(f"unsupported counterfactual type: {counterfactual_type!r}")
    parent = _prepare_parent(parent, counterfactual_type, current_policy)
    sibling = _mutate(parent, counterfactual_type, alternate_policy, current_policy)
    changes = _label_changes(parent.expected_annotations, sibling.expected_annotations)
    if not changes and counterfactual_type != "CAPABILITY_CHANGE":
        raise CounterfactualError("counterfactual mutation did not change any expected label")
    digest = hashlib.sha256(f"{parent.scenario_id}:{counterfactual_type}".encode()).hexdigest()[:20]
    group_id = f"cf_{counterfactual_type.lower()}_{digest}"
    provenance = CounterfactualProvenance(
        counterfactual_group_id=group_id,
        counterfactual_type=counterfactual_type,
        changed_variable=CHANGED_VARIABLES[counterfactual_type],
        parent_scenario_id=parent.scenario_id,
        invariant_fields=_INVARIANTS[counterfactual_type],
        expected_label_changes=changes,
    )
    parent_with_metadata = replace(parent, counterfactual=provenance)
    sibling = replace(sibling, counterfactual=provenance)
    # Validate the closed schema after provenance is attached.
    try:
        parent_with_metadata.require_valid()
        sibling.require_valid()
    except BlueprintError as error:
        raise CounterfactualError(error.errors) from error
    group = CounterfactualGroup(
        group_id=group_id,
        counterfactual_type=counterfactual_type,
        changed_variable=CHANGED_VARIABLES[counterfactual_type],
        parent_scenario_id=parent.scenario_id,
        members=(parent_with_metadata, sibling),
    )
    group.require_valid()
    return group


def make_counterfactual_groups(
    parents: Iterable[ScenarioBlueprint],
    record_counts: Mapping[str, int],
    *,
    alternate_policies: Mapping[str, PolicyDescriptor] | None = None,
    policies: Iterable[PolicyDescriptor] = (),
) -> tuple[CounterfactualGroup, ...]:
    """Create two-record groups from even per-family record quotas.

    A parent is never reused across groups. Policy/delegation changes select only
    catalogue-resolvable alternate policies supplied by the caller.
    """

    ordered = tuple(sorted(parents, key=lambda item: item.scenario_id))
    if not ordered:
        raise CounterfactualError("counterfactual generation requires parent blueprints")
    if set(record_counts) != set(COUNTERFACTUAL_TYPE_ORDER):
        raise CounterfactualError("counterfactual record counts must contain all six families")
    policy_values = tuple(sorted(policies, key=lambda item: (item.domain, item.policy_id)))
    policy_by_id = {policy.policy_id: policy for policy in policy_values}
    result: list[CounterfactualGroup] = []
    used_parent_ids: set[str] = set()
    for type_index, counterfactual_type in enumerate(COUNTERFACTUAL_TYPE_ORDER):
        record_count = record_counts[counterfactual_type]
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 0
            or record_count % 2
        ):
            raise CounterfactualError(f"invalid count for {counterfactual_type}")
        group_count = record_count // 2
        eligible = [
            item
            for item in ordered
            if item.candidate_directive_spec is not None
            and item.expected_annotations.annotation_status == "CONFIRMED"
            and item.scenario_id not in used_parent_ids
        ]
        if not eligible and group_count:
            raise CounterfactualError(f"no eligible parents for {counterfactual_type}")
        ranked = sorted(
            eligible,
            key=lambda item: hashlib.sha256(
                f"{counterfactual_type}:{item.scenario_id}".encode()
            ).digest(),
        )
        if counterfactual_type == "PRESENTATION_CHANGE":
            operative = [
                item
                for item in ranked
                if item.expected_annotations.instruction_presentation == "OPERATIVE"
            ]
            reported = [
                item
                for item in ranked
                if item.expected_annotations.instruction_presentation == "QUOTED_OR_REPORTED"
            ]
            interleaved: list[ScenarioBlueprint] = []
            for index in range(max(len(operative), len(reported))):
                if index < len(operative):
                    interleaved.append(operative[index])
                if index < len(reported):
                    interleaved.append(reported[index])
            ranked = interleaved
        elif counterfactual_type == "POLICY_CHANGE":
            conflicting = [
                item
                for item in ranked
                if item.expected_annotations.protected_policy_alignment == "CONFLICTING"
            ]
            compliant = [
                item
                for item in ranked
                if item.expected_annotations.protected_policy_alignment == "COMPLIANT"
            ]
            interleaved = []
            for index in range(max(len(conflicting), len(compliant))):
                if index < len(conflicting):
                    interleaved.append(conflicting[index])
                if index < len(compliant):
                    interleaved.append(compliant[index])
            ranked = interleaved
        created = 0
        for parent in ranked[type_index:] + ranked[:type_index]:
            if created >= group_count:
                break
            if parent.scenario_id in used_parent_ids:
                continue
            alternate = (alternate_policies or {}).get(parent.policy_id)
            if alternate is None and counterfactual_type == "POLICY_CHANGE":
                candidates = [
                    policy
                    for policy in policy_values
                    if policy.domain == parent.domain
                    and policy.policy_id != parent.policy_id
                    and parent.grouping is not None
                    and policy.policy_family == parent.grouping.policy_family
                ]
                for candidate in candidates:
                    try:
                        group = make_counterfactual_group(
                            parent,
                            counterfactual_type,
                            alternate_policy=candidate,
                            current_policy=policy_by_id.get(parent.policy_id),
                        )
                    except CounterfactualError:
                        continue
                    alternate = candidate
                    break
                else:
                    continue
            try:
                group = make_counterfactual_group(
                    parent,
                    counterfactual_type,
                    alternate_policy=alternate,
                    current_policy=policy_by_id.get(parent.policy_id),
                )
            except CounterfactualError:
                continue
            result.append(group)
            used_parent_ids.add(parent.scenario_id)
            created += 1
        if created != group_count:
            raise CounterfactualError(
                f"could create only {created}/{group_count} groups for {counterfactual_type}"
            )
    return tuple(sorted(result, key=lambda item: item.group_id))


__all__ = [
    "CHANGED_VARIABLES",
    "CounterfactualError",
    "CounterfactualGroup",
    "make_counterfactual_group",
    "make_counterfactual_groups",
    "validate_counterfactual_group",
]
