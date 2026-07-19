from __future__ import annotations

from copy import deepcopy

import pytest

from promptsec.training.dataset import PolicyBenchTorchDataset
from promptsec.training.serialization import (
    NOT_PROVIDED,
    SPECIAL_TOKENS,
    record_sections,
    serialize_full_context,
)
from promptsec.training.token_budget import encode_with_section_budget


def test_full_context_serialization_is_ordered_unicode_and_label_free(record_factory) -> None:
    record = record_factory()
    serialized = serialize_full_context(record)
    positions = [serialized.index(token) for token in SPECIAL_TOKENS[::2]]
    assert positions == sorted(positions)
    assert "naïve café" in serialized
    assert "prompt_injection_verdict" not in serialized
    assert "SILVER_VALIDATED" not in serialized


def test_missing_values_use_stable_marker() -> None:
    sections = record_sections({"context": {}, "content": {}})
    assert sections.protected_policy == NOT_PROVIDED
    assert sections.candidate == NOT_PROVIDED
    assert sections.available_capabilities == NOT_PROVIDED


def test_capabilities_have_deterministic_one_per_line_order(record_factory) -> None:
    record = record_factory()
    assert record_sections(record).available_capabilities == "READ_DOCUMENT\nSEND_EMAIL"


def test_section_budget_preserves_candidate_and_redistributes(
    fake_tokenizer, record_factory
) -> None:
    record = record_factory()
    record["content"]["text"] = " ".join(f"candidate-{index}" for index in range(500))
    result = encode_with_section_budget(
        fake_tokenizer,
        record_sections(record),
        max_length=96,
        head_tail_ratio=0.75,
    )
    stats = result.statistics
    assert len(result.input_ids) <= 96
    assert stats["allocated_content_tokens"]["candidate"] > 0
    assert stats["truncated_sections"]["candidate"]
    assert result.input_ids[-2] == fake_tokenizer.convert_tokens_to_ids("</candidate>")


def test_dataset_does_not_mutate_records(fake_tokenizer, label_mappings, record_factory) -> None:
    record = record_factory()
    original = deepcopy(record)
    dataset = PolicyBenchTorchDataset([record], fake_tokenizer, label_mappings, max_length=96)
    item = dataset[0]
    assert record == original
    assert set(item["labels"]) == set(label_mappings)


def test_too_small_budget_is_rejected(fake_tokenizer, record_factory) -> None:
    with pytest.raises(ValueError, match="at least 32"):
        encode_with_section_budget(fake_tokenizer, record_sections(record_factory()), max_length=24)
