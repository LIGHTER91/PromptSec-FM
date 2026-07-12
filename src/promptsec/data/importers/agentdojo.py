"""Conservative importer for static AgentDojo security-case definitions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from promptsec.data.agentdojo_snapshot import (
    AgentDojoSnapshotError,
    iter_security_cases,
    snapshot_path,
    validate_snapshot,
)
from promptsec.data.importers.base import (
    BaseImporter,
    ImporterError,
    RawRecord,
    portable_path,
)

_MAPPING_CONFIDENCE = 0.6
_MAPPING_VERSION = "agentdojo_mapping_v0.1"
_REVIEW_REASONS = (
    "AgentDojo exposes a static injection-task goal, not the attack string or injection "
    "vector that an execution would place in a tool-visible environment.",
    "Authority, protected-policy alignment, and user-goal alignment require manual review; "
    "no benchmark outcome is imported as annotation ground truth.",
    "AgentDojo task text has no separate upstream data-license declaration and is retained as "
    "a local-only source-derived payload pending review.",
)
_RATIONALE = (
    "SOURCE_DEFINES_INJECTION_TASK_GOAL",
    "STATIC_GOAL_IS_NOT_A_MATERIALIZED_ATTACK_PAYLOAD",
    "NO_RUNTIME_RESULT_USED_AS_GROUND_TRUTH",
    "AUTHORITY_AND_ALIGNMENT_REQUIRE_MANUAL_REVIEW",
)


class AgentDojoImporter(BaseImporter):
    """Enumerate one record per static user-task/injection-task pair."""

    importer_version = "0.1.0"

    def iter_raw_records(self, inputs: Mapping[str, str | Path]) -> Iterator[RawRecord]:
        expected = {artifact.id for artifact in self.config.artifacts}
        unexpected = set(inputs) - expected
        if unexpected:
            raise ImporterError(f"unknown input artifact(s): {sorted(unexpected)}")
        missing = expected - set(inputs)
        if missing:
            raise ImporterError(f"missing input artifact(s): {sorted(missing)}")
        if len(self.config.artifacts) != 1:
            raise ImporterError("AgentDojo config must declare exactly one package artifact")

        artifact = self.config.artifacts[0]
        candidate = snapshot_path(self.config, inputs[artifact.id])
        try:
            snapshot = validate_snapshot(candidate, self.config)
            cases = iter_security_cases(snapshot)
            for index, payload in enumerate(cases):
                yield RawRecord(
                    artifact=artifact,
                    raw_artifact=portable_path(candidate),
                    split=artifact.split,
                    index=index,
                    payload=payload,
                )
        except AgentDojoSnapshotError as exc:
            raise ImporterError(f"cannot import AgentDojo snapshot: {exc}") from exc

    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        goal = _required_string(raw.payload, "injection_goal")
        user_prompt = _required_string(raw.payload, "user_prompt")
        security_case_id = _required_string(raw.payload, "security_case_id")
        suite_id = _required_string(raw.payload, "suite_id")
        benchmark_version = _required_string(raw.payload, "benchmark_version")
        user_task_id = _required_string(raw.payload, "user_task_id")
        injection_task_id = _required_string(raw.payload, "injection_task_id")
        tool_names = raw.payload.get("suite_tool_names")
        if (
            not isinstance(tool_names, list)
            or not tool_names
            or any(not isinstance(item, str) or not item for item in tool_names)
        ):
            raise ImporterError("AgentDojo suite_tool_names must be a non-empty string array")

        annotations = {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": "UNDETERMINED",
            "protected_policy_alignment": "UNDETERMINED",
            "authority_status": "UNKNOWN",
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": [],
            "spans": [{"start": 0, "end": len(goal), "type": "DIRECTIVE"}],
            "annotation_status": "INSUFFICIENT_CONTEXT",
            "annotator_confidence": _MAPPING_CONFIDENCE,
        }
        original_labels = {
            "benchmark_version": benchmark_version,
            "injection_task_difficulty": _required_string(raw.payload, "injection_task_difficulty"),
            "injection_task_id": injection_task_id,
            "source_definition_kind": "STATIC_USER_TASK_INJECTION_TASK_PAIR",
            "suite_id": suite_id,
            "user_task_difficulty": _required_string(raw.payload, "user_task_difficulty"),
            "user_task_id": user_task_id,
        }
        field_mappings = [
            {
                "source": "injection_goal",
                "target": "content.text",
                "method": "COPY_EXACT_STATIC_INJECTION_GOAL_DEFINITION",
            },
            {
                "source": "user_prompt",
                "target": "context.user_goal",
                "method": "COPY_EXACT_USER_TASK_PROMPT",
            },
            {
                "source": "suite_tool_names",
                "target": "context.available_capabilities",
                "method": "COPY_SORTED_SUITE_CAPABILITY_NAMES",
            },
            {
                "source": "injection_task_id",
                "target": "annotations.attack_families",
                "method": "SOURCE_INJECTION_TASK_MEMBERSHIP_TO_PROMPT_INJECTION_NEEDS_REVIEW",
            },
        ]
        record = self.build_record(
            raw,
            text=goal,
            source_text=goal,
            annotations=annotations,
            original_labels=original_labels,
            mapping_status="NEEDS_REVIEW",
            field_mappings=field_mappings,
            review_reasons=_REVIEW_REASONS,
            transformations=(
                "ENUMERATE_STATIC_SECURITY_CASE_CARTESIAN_PRODUCT",
                "PRESERVE_INJECTION_GOAL_WITHOUT_ATTACK_EXECUTION",
            ),
            context={
                "protected_policy": None,
                "user_goal": user_prompt,
                "available_capabilities": list(tool_names),
            },
        )
        template_family = f"agentdojo:{benchmark_version}:{suite_id}:{injection_task_id}"
        record["extensions"] = {
            "agentic_source": {
                "agentic_group_id": security_case_id,
                "attack_configuration": None,
                "attack_name": None,
                "benchmark_version": benchmark_version,
                "effective_benchmark_version": _required_string(
                    raw.payload, "effective_benchmark_version"
                ),
                "initial_state_reference": _required_string(raw.payload, "initial_state_reference"),
                "injection_bearing_content_status": ("not_materialized_static_definition"),
                "injection_task_id": injection_task_id,
                "injection_vectors_reference": _required_string(
                    raw.payload, "injection_vectors_reference"
                ),
                "runtime_observations_status": "not_imported",
                "security_case_id": security_case_id,
                "security_case_id_origin": "derived_from_upstream_task_pair",
                "source_definition_type": "static_security_case_definition",
                "source_locations": {
                    "injection_task_class": _required_string(
                        raw.payload, "injection_task_definition_class"
                    ),
                    "injection_task_module": _required_string(
                        raw.payload, "injection_task_definition_module"
                    ),
                    "user_task_class": _required_string(raw.payload, "user_task_definition_class"),
                    "user_task_module": _required_string(
                        raw.payload, "user_task_definition_module"
                    ),
                },
                "suite_id": suite_id,
                "suite_tool_names": list(tool_names),
                "targeted_tool_names": [],
                "targeted_tools_status": "not_enumerated_without_task_execution",
                "template_family": template_family,
                "upstream_revision": self.config.revision,
                "upstream_version": self.config.version,
                "user_task_id": user_task_id,
            },
            "mapping_evidence": {
                "label_origin": "SOURCE_DERIVED",
                "mapping_confidence": _MAPPING_CONFIDENCE,
                "mapping_version": _MAPPING_VERSION,
                "rationale": list(_RATIONALE),
                "requires_manual_review": True,
            },
        }
        yield record


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ImporterError(f"AgentDojo field {field!r} must be a non-empty string")
    return value


__all__ = ["AgentDojoImporter"]
