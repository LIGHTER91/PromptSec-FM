"""Exact character-span construction over unmodified PolicyBench candidate text."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

SpanType = Literal["DIRECTIVE", "INJECTION_PAYLOAD", "AUTHORITY_CLAIM"]
SPAN_TYPES: tuple[SpanType, ...] = (
    "DIRECTIVE",
    "INJECTION_PAYLOAD",
    "AUTHORITY_CLAIM",
)


class SpanError(ValueError):
    """Raised when an explicit anchor cannot produce an exact canonical span."""

    def __init__(self, errors: str | Sequence[str]) -> None:
        self.errors = (errors,) if isinstance(errors, str) else tuple(errors)
        if not self.errors:
            raise ValueError("SpanError requires at least one error")
        super().__init__(
            "PolicyBench span validation failed:\n"
            + "\n".join(f"- {error}" for error in self.errors)
        )


def _occurrence(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpanError(f"{context}.occurrence must be a zero-based non-negative integer")
    return value


def _span_type(value: Any, context: str) -> SpanType:
    if value not in SPAN_TYPES:
        raise SpanError(f"{context}.type must be one of {list(SPAN_TYPES)}")
    return value


@dataclass(frozen=True, slots=True)
class SpanAnchor:
    """An exact substring and its zero-based occurrence in the final text.

    Occurrences are found one Python Unicode code point after the previous match,
    so repeated and overlapping substring occurrences remain distinguishable.
    """

    text: str
    occurrence: int = 0
    span_type: SpanType = "DIRECTIVE"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise SpanError("anchor.text must be a non-empty string")
        _occurrence(self.occurrence, "anchor")
        _span_type(self.span_type, "anchor")

    def to_dict(self, *, include_type: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {"text": self.text, "occurrence": self.occurrence}
        if include_type:
            value["type"] = self.span_type
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        span_type: SpanType | None = None,
        context: str = "anchor",
    ) -> SpanAnchor:
        if not isinstance(value, Mapping):
            raise SpanError(f"{context} must be an object")
        expected = {"text", "occurrence"} | ({"type"} if span_type is None else set())
        if set(value) != expected:
            raise SpanError(f"{context} must contain exactly {sorted(expected)}")
        text = value.get("text")
        if not isinstance(text, str) or not text:
            raise SpanError(f"{context}.text must be a non-empty string")
        resolved_type = _span_type(value.get("type"), context) if span_type is None else span_type
        return cls(
            text=text,
            occurrence=_occurrence(value.get("occurrence"), context),
            span_type=resolved_type,
        )


@dataclass(frozen=True, slots=True, order=True)
class CanonicalSpan:
    start: int
    end: int
    span_type: SpanType

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise SpanError("canonical span must satisfy 0 <= start < end")
        _span_type(self.span_type, "span")

    def to_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "type": self.span_type}

    @classmethod
    def from_mapping(cls, value: Any, *, context: str = "span") -> CanonicalSpan:
        if not isinstance(value, Mapping) or set(value) != {"start", "end", "type"}:
            raise SpanError(f"{context} must contain exactly start, end, and type")
        start, end = value.get("start"), value.get("end")
        if isinstance(start, bool) or not isinstance(start, int):
            raise SpanError(f"{context}.start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise SpanError(f"{context}.end must be an integer")
        return cls(start=start, end=end, span_type=_span_type(value.get("type"), context))


def occurrence_start(text: str, substring: str, occurrence: int = 0) -> int:
    """Return an exact zero-based occurrence without normalizing either string."""

    if not isinstance(text, str):
        raise SpanError("candidate text must be a string")
    if not isinstance(substring, str) or not substring:
        raise SpanError("anchor text must be a non-empty string")
    wanted = _occurrence(occurrence, "anchor")
    start = -1
    search_from = 0
    for _ in range(wanted + 1):
        start = text.find(substring, search_from)
        if start < 0:
            raise SpanError(
                f"anchor {substring!r} occurrence {wanted} is not present in candidate text"
            )
        search_from = start + 1
    return start


def resolve_anchor(text: str, anchor: SpanAnchor) -> CanonicalSpan:
    start = occurrence_start(text, anchor.text, anchor.occurrence)
    return CanonicalSpan(
        start=start,
        end=start + len(anchor.text),
        span_type=anchor.span_type,
    )


def resolve_anchors(
    text: str,
    anchors: Iterable[SpanAnchor],
    *,
    sort: bool = True,
) -> tuple[CanonicalSpan, ...]:
    """Resolve anchors while retaining legitimate overlaps between span types."""

    resolved: list[CanonicalSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, SpanAnchor):
            raise SpanError(f"anchors[{index}] must be a SpanAnchor")
        span = resolve_anchor(text, anchor)
        identity = (span.start, span.end, span.span_type)
        if identity in seen:
            raise SpanError(f"anchors[{index}] duplicates an existing canonical span")
        seen.add(identity)
        resolved.append(span)
    if sort:
        resolved.sort(key=lambda item: (item.start, item.end, SPAN_TYPES.index(item.span_type)))
    return tuple(resolved)


def resolve_generation_anchors(
    candidate_text: str,
    *,
    directive_anchors: Iterable[Mapping[str, Any]] = (),
    injection_payload_anchors: Iterable[Mapping[str, Any]] = (),
    authority_claim_anchors: Iterable[Mapping[str, Any]] = (),
) -> tuple[CanonicalSpan, ...]:
    """Resolve the three closed anchor arrays emitted by a generation provider."""

    typed: list[SpanAnchor] = []
    for values, span_type, name in (
        (directive_anchors, "DIRECTIVE", "directive_anchors"),
        (injection_payload_anchors, "INJECTION_PAYLOAD", "injection_payload_anchors"),
        (authority_claim_anchors, "AUTHORITY_CLAIM", "authority_claim_anchors"),
    ):
        for index, value in enumerate(values):
            typed.append(
                SpanAnchor.from_mapping(
                    value,
                    span_type=span_type,
                    context=f"{name}[{index}]",
                )
            )
    return resolve_anchors(candidate_text, typed)


def validate_spans(text: Any, spans: Any) -> list[str]:
    """Return deterministic structural errors for canonical half-open spans."""

    if not isinstance(text, str):
        return ["candidate text must be a string"]
    if not isinstance(spans, (list, tuple)):
        return ["spans must be an array"]
    errors: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    for index, value in enumerate(spans):
        try:
            span = (
                value
                if isinstance(value, CanonicalSpan)
                else CanonicalSpan.from_mapping(value, context=f"spans[{index}]")
            )
        except SpanError as error:
            errors.extend(error.errors)
            continue
        if span.end > len(text):
            errors.append(f"spans[{index}] ends at {span.end}, beyond candidate length {len(text)}")
        identity = (span.start, span.end, span.span_type)
        if identity in seen:
            errors.append(f"spans[{index}] duplicates an earlier span")
        seen.add(identity)
    return errors


def require_valid_spans(text: str, spans: Any) -> None:
    errors = validate_spans(text, spans)
    if errors:
        raise SpanError(errors)


__all__ = [
    "CanonicalSpan",
    "SPAN_TYPES",
    "SpanAnchor",
    "SpanError",
    "SpanType",
    "occurrence_start",
    "require_valid_spans",
    "resolve_anchor",
    "resolve_anchors",
    "resolve_generation_anchors",
    "validate_spans",
]
