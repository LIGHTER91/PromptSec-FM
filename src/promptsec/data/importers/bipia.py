"""Importer for BIPIA's category-to-payload attack JSON files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from promptsec.data.importers.base import BaseImporter, ImporterError, RawRecord, portable_path

_MAPPING_REVIEW_REASON = (
    "BIPIA confirms an injected task, supporting TASK_HIJACKING, but its attack-domain "
    "categories do not justify more specific PromptSec-FM objectives."
)
_LICENSE_REVIEW_REASON = (
    "Redistribution terms for BIPIA attack JSON are NOASSERTION and require release-time "
    "license review."
)


class BIPIAImporter(BaseImporter):
    """Flatten each BIPIA attack category into five provenance-preserving records."""

    def iter_raw_records(self, inputs: Mapping[str, str | Path]) -> Iterator[RawRecord]:
        resolved_inputs = self._validated_inputs(inputs)
        for artifact in self.config.artifacts:
            source_path = resolved_inputs[artifact.id]
            payload_type = self._payload_type(artifact.id)
            categories = self._read_categories(source_path, encoding=artifact.encoding)
            raw_index = 0
            for attack_domain, instructions in categories.items():
                for domain_index, instruction in enumerate(instructions):
                    yield RawRecord(
                        artifact=artifact,
                        raw_artifact=portable_path(source_path),
                        split=artifact.split,
                        index=raw_index,
                        payload={
                            "source_id": f"{artifact.id}:{attack_domain}:{domain_index}",
                            "attack_domain": attack_domain,
                            "payload_type": payload_type,
                            "instruction": instruction,
                        },
                    )
                    raw_index += 1

    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        instruction = raw.payload.get("instruction")
        attack_domain = raw.payload.get("attack_domain")
        payload_type = raw.payload.get("payload_type")
        if not isinstance(instruction, str) or not instruction:
            raise ImporterError("BIPIA instruction must be a non-empty string")
        if not isinstance(attack_domain, str) or not attack_domain:
            raise ImporterError("BIPIA attack_domain must be a non-empty string")
        if payload_type not in {"text", "code"}:
            raise ImporterError(f"unsupported BIPIA payload_type: {payload_type!r}")

        annotations = {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": "MISALIGNED",
            "protected_policy_alignment": "UNDETERMINED",
            "authority_status": "OUTSIDE_AUTHORITY",
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": ["TASK_HIJACKING"],
            "spans": [
                {
                    "start": 0,
                    "end": len(instruction),
                    "type": "INJECTION_PAYLOAD",
                }
            ],
            "annotation_status": "CONFIRMED",
            "annotator_confidence": 0.9,
        }
        field_mappings = [
            {
                "source": "instruction",
                "target": "content.text",
                "method": "COPY_EXACT",
            },
            {
                "source": "payload_type",
                "target": "content.modality",
                "method": "CODE_TO_CODE_OTHERWISE_TEXT",
            },
            {
                "source": "source_id",
                "target": "annotations.attack_families",
                "method": "BIPIA_BENCHMARK_MEMBERSHIP_TO_PROMPT_INJECTION_NEEDS_REVIEW",
            },
            {
                "source": "source_id",
                "target": "annotations.attack_objectives",
                "method": "BIPIA_INJECTED_TASK_ROLE_TO_TASK_HIJACKING_NEEDS_REVIEW",
            },
        ]

        yield self.build_record(
            raw,
            text=instruction,
            source_text=instruction,
            annotations=annotations,
            original_labels={
                "attack_domain": attack_domain,
                "payload_type": payload_type,
            },
            mapping_status="NEEDS_REVIEW",
            field_mappings=field_mappings,
            unmapped_labels=[attack_domain],
            review_reasons=[_MAPPING_REVIEW_REASON, _LICENSE_REVIEW_REASON],
            content_overrides={
                "content_origin": "CODE_REPOSITORY" if payload_type == "code" else "UNKNOWN",
                "modality": "CODE" if payload_type == "code" else "TEXT",
            },
        )

    def _validated_inputs(self, inputs: Mapping[str, str | Path]) -> dict[str, Path]:
        expected = {artifact.id for artifact in self.config.artifacts}
        unexpected = set(inputs) - expected
        if unexpected:
            raise ImporterError(f"unknown input artifact(s): {sorted(unexpected)}")
        missing = expected - set(inputs)
        if missing:
            raise ImporterError(f"missing input artifact(s): {sorted(missing)}")
        return {artifact_id: Path(path) for artifact_id, path in inputs.items()}

    @staticmethod
    def _payload_type(artifact_id: str) -> str:
        if artifact_id.startswith("text_attack_"):
            return "text"
        if artifact_id.startswith("code_attack_"):
            return "code"
        raise ImporterError(f"cannot determine BIPIA payload type from artifact {artifact_id!r}")

    @staticmethod
    def _read_categories(path: Path, *, encoding: str) -> dict[str, list[str]]:
        try:
            value = json.loads(path.read_text(encoding=encoding))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ImporterError(f"cannot read BIPIA artifact {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ImporterError(f"{path}: expected a category-to-list JSON object")

        categories: dict[str, list[str]] = {}
        for attack_domain, instructions in value.items():
            if not isinstance(attack_domain, str) or not attack_domain:
                raise ImporterError(f"{path}: category names must be non-empty strings")
            if not isinstance(instructions, list) or len(instructions) != 5:
                raise ImporterError(
                    f"{path}: category {attack_domain!r} must contain exactly five payloads"
                )
            if not all(isinstance(item, str) and item for item in instructions):
                raise ImporterError(
                    f"{path}: category {attack_domain!r} payloads must be non-empty strings"
                )
            categories[attack_domain] = instructions
        return categories


__all__ = ["BIPIAImporter"]
