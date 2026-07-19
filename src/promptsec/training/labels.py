"""Schema-derived deterministic mappings for all nine multi-task heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_json
from promptsec.policybench.io import read_json_object

SINGLE_LABEL_HEADS = (
    "prompt_injection_verdict",
    "instruction_presence",
    "instruction_presentation",
    "instruction_addressee",
    "user_goal_alignment",
    "protected_policy_alignment",
    "authority_status",
)
MULTILABEL_HEADS = ("attack_families", "attack_objectives")
HEADS = (*SINGLE_LABEL_HEADS, *MULTILABEL_HEADS)


@dataclass(frozen=True, slots=True)
class LabelMapping:
    head: str
    labels: tuple[str, ...]
    multilabel: bool
    taxonomy_version: str = "1.0"

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.labels)}

    @property
    def mapping_hash(self) -> str:
        return sha256_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["labels"] = list(self.labels)
        value["label_to_id"] = self.label_to_id
        value["id_to_label"] = {str(key): item for key, item in self.id_to_label.items()}
        if include_hash:
            value["mapping_hash"] = self.mapping_hash
        return value


def load_label_mappings(schema_path: str | Path) -> dict[str, LabelMapping]:
    schema = read_json_object(schema_path)
    annotations = schema["properties"]["annotations"]["properties"]
    derived = schema["properties"]["derived"]["properties"]
    mappings: dict[str, LabelMapping] = {}
    for head in SINGLE_LABEL_HEADS:
        values = (
            derived[head]["enum"]
            if head == "prompt_injection_verdict"
            else annotations[head]["enum"]
        )
        mappings[head] = LabelMapping(head, tuple(values), False)
    for head in MULTILABEL_HEADS:
        mappings[head] = LabelMapping(head, tuple(annotations[head]["items"]["enum"]), True)
    return mappings


def _head_value(record: Mapping[str, Any], head: str) -> Any:
    parent = "derived" if head == "prompt_injection_verdict" else "annotations"
    container = record.get(parent)
    return container.get(head) if isinstance(container, Mapping) else None


def encode_labels(
    record: Mapping[str, Any], mappings: Mapping[str, LabelMapping]
) -> dict[str, int | list[float]]:
    encoded: dict[str, int | list[float]] = {}
    for head, mapping in mappings.items():
        value = _head_value(record, head)
        if mapping.multilabel:
            if not isinstance(value, list):
                raise ValueError(f"record {record.get('id')!r}: {head} must be an array")
            unknown = set(value).difference(mapping.labels)
            if unknown:
                raise ValueError(f"record {record.get('id')!r}: unknown {head}: {sorted(unknown)}")
            selected = set(value)
            encoded[head] = [float(label in selected) for label in mapping.labels]
        else:
            if value not in mapping.label_to_id:
                raise ValueError(f"record {record.get('id')!r}: unknown {head}: {value!r}")
            encoded[head] = mapping.label_to_id[value]
    return encoded


def mappings_fingerprint(mappings: Mapping[str, LabelMapping]) -> str:
    return sha256_json({head: mapping.to_dict() for head, mapping in mappings.items()})


def write_label_mappings(path: str | Path, mappings: Mapping[str, LabelMapping]) -> None:
    from promptsec.policybench.io import write_json

    write_json(
        path,
        {
            "taxonomy_version": "1.0",
            "mapping_hash": mappings_fingerprint(mappings),
            "heads": {head: mapping.to_dict() for head, mapping in mappings.items()},
        },
    )
