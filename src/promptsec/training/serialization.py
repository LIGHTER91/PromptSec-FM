"""Deterministic full-context serialization without label leakage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NOT_PROVIDED = "[NOT_PROVIDED]"
SPECIAL_TOKENS = (
    "<protected_policy>",
    "</protected_policy>",
    "<user_goal>",
    "</user_goal>",
    "<source>",
    "</source>",
    "<available_capabilities>",
    "</available_capabilities>",
    "<candidate>",
    "</candidate>",
)


@dataclass(frozen=True, slots=True)
class ContextSections:
    protected_policy: str
    user_goal: str
    source: str
    available_capabilities: str
    candidate: str

    def as_ordered_items(self) -> tuple[tuple[str, str], ...]:
        return (
            ("protected_policy", self.protected_policy),
            ("user_goal", self.user_goal),
            ("source", self.source),
            ("available_capabilities", self.available_capabilities),
            ("candidate", self.candidate),
        )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else NOT_PROVIDED


def _capabilities(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return NOT_PROVIDED
    return "\n".join(item if isinstance(item, str) else NOT_PROVIDED for item in value)


def record_sections(record: Mapping[str, Any]) -> ContextSections:
    context = record.get("context")
    content = record.get("content")
    context = context if isinstance(context, Mapping) else {}
    content = content if isinstance(content, Mapping) else {}
    source = "\n".join(
        (
            f"source_role={_text(content.get('source_role'))}",
            f"content_origin={_text(content.get('content_origin'))}",
            f"delivery_mode={_text(content.get('delivery_mode'))}",
            f"ingestion_path={_text(content.get('ingestion_path'))}",
            f"modality={_text(content.get('modality'))}",
            f"source_integrity={_text(content.get('source_integrity'))}",
        )
    )
    return ContextSections(
        protected_policy=_text(context.get("protected_policy")),
        user_goal=_text(context.get("user_goal")),
        source=source,
        available_capabilities=_capabilities(context.get("available_capabilities")),
        candidate=_text(content.get("text")),
    )


def serialize_sections(sections: ContextSections) -> str:
    blocks = []
    for name, value in sections.as_ordered_items():
        blocks.append(f"<{name}>\n{value}\n</{name}>")
    return "\n\n".join(blocks)


def serialize_full_context(record: Mapping[str, Any]) -> str:
    return serialize_sections(record_sections(record))
