"""Section-aware token allocation that preserves the candidate section."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from promptsec.training.serialization import ContextSections

SECTION_ORDER = (
    "protected_policy",
    "user_goal",
    "source",
    "available_capabilities",
    "candidate",
)
NOMINAL_BUDGETS = {
    "protected_policy": 120,
    "user_goal": 64,
    "source": 36,
    "available_capabilities": 36,
    "candidate": 240,
}
REDISTRIBUTION_ORDER = (
    "candidate",
    "protected_policy",
    "user_goal",
    "source",
    "available_capabilities",
)


@dataclass(frozen=True, slots=True)
class TokenBudgetResult:
    input_ids: list[int]
    attention_mask: list[int]
    statistics: dict[str, Any]


def _truncate_head_tail(values: list[int], limit: int, head_ratio: float) -> list[int]:
    if len(values) <= limit:
        return list(values)
    if limit <= 0:
        return []
    if limit == 1:
        return values[:1]
    head = min(limit - 1, max(1, math.floor(limit * head_ratio)))
    tail = limit - head
    return values[:head] + values[-tail:]


def _initial_budgets(capacity: int, lengths: Mapping[str, int]) -> dict[str, int]:
    total = sum(NOMINAL_BUDGETS.values())
    budgets = {
        name: max(1 if lengths[name] else 0, math.floor(capacity * NOMINAL_BUDGETS[name] / total))
        for name in SECTION_ORDER
    }
    while sum(budgets.values()) > capacity:
        for name in reversed(REDISTRIBUTION_ORDER):
            minimum = 1 if lengths[name] else 0
            if budgets[name] > minimum:
                budgets[name] -= 1
                break
        else:
            raise ValueError("max_length is too small to retain every non-empty section")
    remainder = capacity - sum(budgets.values())
    for name in REDISTRIBUTION_ORDER:
        if remainder <= 0:
            break
        desired = NOMINAL_BUDGETS[name] - budgets[name]
        addition = min(remainder, max(0, desired))
        budgets[name] += addition
        remainder -= addition
    if remainder:
        budgets["candidate"] += remainder
    return budgets


def encode_with_section_budget(
    tokenizer: Any,
    sections: ContextSections,
    *,
    max_length: int = 512,
    head_tail_ratio: float = 0.75,
) -> TokenBudgetResult:
    """Tokenize sections separately, redistribute unused budget, then add model tokens."""

    if max_length < 32:
        raise ValueError("max_length must be at least 32")
    section_text = dict(sections.as_ordered_items())
    tokenized = {
        name: list(tokenizer.encode(section_text[name], add_special_tokens=False))
        for name in SECTION_ORDER
    }
    marker_count = len(SECTION_ORDER) * 2
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = max_length - marker_count - special_count
    if capacity < sum(bool(values) for values in tokenized.values()):
        raise ValueError("max_length cannot preserve all non-empty sections and markers")
    lengths = {name: len(values) for name, values in tokenized.items()}
    budgets = _initial_budgets(capacity, lengths)
    allocated = {name: min(lengths[name], budgets[name]) for name in SECTION_ORDER}
    unused = capacity - sum(allocated.values())
    for name in REDISTRIBUTION_ORDER:
        if unused <= 0:
            break
        addition = min(unused, lengths[name] - allocated[name])
        allocated[name] += addition
        unused -= addition
    content_ids: list[int] = []
    truncated: dict[str, bool] = {}
    for name in SECTION_ORDER:
        opening = tokenizer.convert_tokens_to_ids(f"<{name}>")
        closing = tokenizer.convert_tokens_to_ids(f"</{name}>")
        if opening is None or closing is None:
            raise ValueError(f"tokenizer is missing section markers for {name}")
        values = _truncate_head_tail(tokenized[name], allocated[name], head_tail_ratio)
        if name == "candidate" and tokenized[name] and not values:
            raise ValueError("candidate token preservation failed")
        truncated[name] = len(values) < lengths[name]
        content_ids.extend((int(opening), *values, int(closing)))
    input_ids = list(tokenizer.build_inputs_with_special_tokens(content_ids))
    if len(input_ids) > max_length:
        raise ValueError("section-aware allocation exceeded max_length")
    return TokenBudgetResult(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        statistics={
            "max_length": max_length,
            "encoded_length": len(input_ids),
            "original_token_lengths": lengths,
            "allocated_content_tokens": allocated,
            "truncated_sections": truncated,
            "record_truncated": any(truncated.values()),
        },
    )
