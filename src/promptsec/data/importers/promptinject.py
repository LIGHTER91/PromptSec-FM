"""Safe importer for the PromptInject attack-template collections."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from promptsec.data.importers.base import BaseImporter, ImporterError, RawRecord, portable_path

_COLLECTION_OBJECTIVES = {
    "goal_hikacking_attacks": ("GOAL_HIJACKING", "TASK_HIJACKING"),
    "prompt_leaking_attacks": (
        "PROMPT_LEAKING",
        "PROMPT_OR_POLICY_DISCLOSURE",
    ),
}
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


class PromptInjectImporter(BaseImporter):
    """Import only the two upstream attack dictionaries without executing Python."""

    def iter_raw_records(self, inputs: Mapping[str, str | Path]) -> Iterator[RawRecord]:
        resolved_inputs = self._validated_inputs(inputs)
        index = 0
        for artifact in self.config.artifacts:
            source_path = resolved_inputs[artifact.id]
            collections = self._read_attack_collections(source_path, encoding=artifact.encoding)
            for collection in _COLLECTION_OBJECTIVES:
                for key, template in collections[collection].items():
                    label = template.get("label")
                    instruction = template.get("instruction")
                    if not isinstance(label, str) or not label:
                        raise ImporterError(
                            f"{source_path}: {collection}.{key}.label must be a non-empty string"
                        )
                    if not isinstance(instruction, str) or not instruction:
                        raise ImporterError(
                            f"{source_path}: {collection}.{key}.instruction must be "
                            "a non-empty string"
                        )
                    placeholders = list(dict.fromkeys(_PLACEHOLDER_RE.findall(instruction)))
                    yield RawRecord(
                        artifact=artifact,
                        raw_artifact=portable_path(source_path),
                        split=artifact.split,
                        index=index,
                        payload={
                            "source_id": f"{collection}:{key}",
                            "collection": collection,
                            "key": key,
                            "label": label,
                            "instruction": instruction,
                            "placeholders": placeholders,
                        },
                    )
                    index += 1

    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        collection = raw.payload.get("collection")
        if collection not in _COLLECTION_OBJECTIVES:
            raise ImporterError(f"unsupported PromptInject collection: {collection!r}")
        instruction = raw.payload.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ImporterError("PromptInject instruction must be a non-empty string")

        legacy_objective, objective = _COLLECTION_OBJECTIVES[collection]
        annotations = {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": "MISALIGNED",
            "protected_policy_alignment": "UNDETERMINED",
            "authority_status": "OUTSIDE_AUTHORITY",
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": [objective],
            "spans": [
                {
                    "start": 0,
                    "end": len(instruction),
                    "type": "INJECTION_PAYLOAD",
                }
            ],
            "annotation_status": "CONFIRMED",
            "annotator_confidence": 1.0,
        }
        original_labels = {field: raw.payload[field] for field in ("collection", "key", "label")}
        yield self.build_record(
            raw,
            text=instruction,
            source_text=instruction,
            annotations=annotations,
            original_labels=original_labels,
            mapping_status="DETERMINISTIC",
            field_mappings=[
                {
                    "source": "instruction",
                    "target": "content.text",
                    "method": "COPY_EXACT",
                },
                {
                    "source": "collection",
                    "target": "annotations.attack_objectives",
                    "method": f"{legacy_objective}_TO_{objective}",
                },
                {
                    "source": "collection",
                    "target": "annotations.attack_families",
                    "method": "PROMPTINJECT_ATTACK_COLLECTION_TO_PROMPT_INJECTION",
                },
            ],
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

    def _read_attack_collections(
        self, path: Path, *, encoding: str
    ) -> dict[str, dict[str, dict[str, Any]]]:
        try:
            source = path.read_text(encoding=encoding)
            module = ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ImporterError(f"cannot parse PromptInject artifact {path}: {exc}") from exc

        values: dict[str, Any] = {}
        for statement in module.body:
            target_name: str | None = None
            value_node: ast.expr | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    target_name = target.id
                    value_node = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                target_name = statement.target.id
                value_node = statement.value

            if target_name not in _COLLECTION_OBJECTIVES or value_node is None:
                continue
            if target_name in values:
                raise ImporterError(f"{path}: duplicate assignment for {target_name}")
            try:
                values[target_name] = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
                raise ImporterError(
                    f"{path}: {target_name} must be an AST-literal dictionary"
                ) from exc

        missing = set(_COLLECTION_OBJECTIVES) - set(values)
        if missing:
            raise ImporterError(f"{path}: missing attack collection(s): {sorted(missing)}")

        collections: dict[str, dict[str, dict[str, Any]]] = {}
        for name in _COLLECTION_OBJECTIVES:
            value = values[name]
            if not isinstance(value, dict):
                raise ImporterError(f"{path}: {name} must be a dictionary")
            collection: dict[str, dict[str, Any]] = {}
            for key, template in value.items():
                if not isinstance(key, str) or not key:
                    raise ImporterError(f"{path}: {name} keys must be non-empty strings")
                if not isinstance(template, dict):
                    raise ImporterError(f"{path}: {name}.{key} must be a dictionary")
                collection[key] = template
            collections[name] = collection
        return collections


__all__ = ["PromptInjectImporter"]
