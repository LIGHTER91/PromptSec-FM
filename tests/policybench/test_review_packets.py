from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from promptsec.data.hashing import sha256_file
from promptsec.policybench.review_packets import (
    PolicyBenchReviewError,
    create_review_packets,
    select_review_candidates,
)
from tests.policybench.test_quality import _record


def _review_records() -> list[dict]:
    records = [
        _record("bank_en", "Banking fact in English."),
        _record("bank_fr", "Fait bancaire en français.", language="fr"),
        _record("email_en", "Email fact.", domain="email"),
        _record("calendar_fr", "Fait calendrier.", domain="calendar", language="fr"),
        _record("file_en", "File fact.", domain="file_management"),
        _record("web_fr", "Fait achat.", domain="web_and_purchases", language="fr"),
        _record(
            "cf_a",
            "Counterfactual context A.",
            domain="persistent_memory",
            counterfactual_group="cf_review_pair",
        ),
        _record(
            "cf_b",
            "Counterfactual context B.",
            domain="persistent_memory",
            counterfactual_group="cf_review_pair",
        ),
    ]
    for record in records:
        record["extensions"]["policybench_v0_1"]["dataset_split"] = "human_review_candidates"
    return records


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_candidate_selection_is_input_order_independent_and_group_atomic() -> None:
    records = _review_records()

    forward, forward_report = select_review_candidates(records, record_count=5, seed=31)
    reverse, reverse_report = select_review_candidates(reversed(records), record_count=5, seed=31)

    assert [record["id"] for record in forward] == [record["id"] for record in reverse]
    assert forward_report == reverse_report
    selected = {record["id"] for record in forward}
    assert selected.isdisjoint({"cf_a", "cf_b"}) or {"cf_a", "cf_b"}.issubset(selected)
    assert forward_report["gold_claim_permitted"] is False


def test_candidate_selection_expands_when_designated_pool_is_too_small() -> None:
    records = _review_records()
    for record in records[2:]:
        record["extensions"]["policybench_v0_1"]["dataset_split"] = "train"

    selected, report = select_review_candidates(records, record_count=4, seed=31)

    assert len(selected) == 4
    assert report["eligible_pool"] == "all_validated_silver_records"
    assert report["eligible_records"] == len(records)


def test_ordinary_records_in_one_large_split_group_remain_independently_selectable() -> None:
    records = _review_records()[:3]
    for record in records:
        record["extensions"]["policybench_v0_1"]["grouping"]["split_group_id"] = (
            "split_component_with_hundreds_of_records"
        )

    selected, report = select_review_candidates(records, record_count=2, seed=11)

    assert len(selected) == 2
    assert report["selected_records"] == report["requested_records"] == 2
    assert report["deficiencies"] == []


def test_packets_are_blinded_independently_shuffled_and_checksummed(tmp_path: Path) -> None:
    records = _review_records()

    result = create_review_packets(
        records,
        tmp_path,
        record_count=len(records),
        seed=20260715,
    )

    annotator_a = _read_jsonl(tmp_path / "annotator_A.jsonl")
    annotator_b = _read_jsonl(tmp_path / "annotator_B.jsonl")
    assert {item["blind_id"] for item in annotator_a} == {item["blind_id"] for item in annotator_b}
    assert [item["blind_id"] for item in annotator_a] != [item["blind_id"] for item in annotator_b]
    assert result.packet_manifest["automatic_gold_records"] == 0
    assert result.packet_manifest["human_validation_status"] == "PENDING"

    packet = annotator_a[0]
    assert set(packet) == {"blind_id", "content", "context", "annotation", "notes"}
    assert set(packet["context"]) == {
        "protected_policy",
        "user_goal",
        "available_capabilities",
    }
    assert set(packet["content"]) == {
        "text",
        "language",
        "delivery_mode",
        "source_role",
        "content_origin",
        "ingestion_path",
        "modality",
        "source_integrity",
    }
    assert packet["annotation"]["instruction_presence"] is None
    assert packet["annotation"]["attack_families"] == []
    assert packet["annotation"]["spans"] == []
    serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    for hidden in (
        "expected_annotations",
        "blueprint",
        "generator",
        "failed_attempts",
        "validation",
        "counterfactual",
        "metadata",
        "dataset_provenance",
    ):
        assert hidden not in serialized

    expected_files = {
        "annotator_A.jsonl",
        "annotator_B.jsonl",
        "researcher_manifest.json",
        "packet_manifest.json",
        "selection_report.json",
    }
    checksum_lines = (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    observed = {}
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        observed[name] = digest
    assert set(observed) == expected_files
    assert all(observed[name] == sha256_file(tmp_path / name) for name in expected_files)

    researcher = json.loads((tmp_path / "researcher_manifest.json").read_text(encoding="utf-8"))
    assert {item["canonical_id"] for item in researcher["records"]} == {
        record["id"] for record in records
    }
    assert all(item["human_validation_status"] == "PENDING" for item in researcher["records"])


def test_packet_build_is_byte_reproducible(tmp_path: Path) -> None:
    records = _review_records()
    first = tmp_path / "first"
    second = tmp_path / "second"

    create_review_packets(records, first, record_count=len(records), seed=77)
    create_review_packets(reversed(records), second, record_count=len(records), seed=77)

    names = {
        "annotator_A.jsonl",
        "annotator_B.jsonl",
        "researcher_manifest.json",
        "packet_manifest.json",
        "selection_report.json",
        "checksums.sha256",
    }
    assert {name: (first / name).read_bytes() for name in names} == {
        name: (second / name).read_bytes() for name in names
    }


def test_review_packets_reject_any_automatic_gold_claim() -> None:
    record = copy.deepcopy(_review_records()[0])
    record["extensions"]["policybench_v0_1"]["data_quality"] = "GOLD_HUMAN_CONFIRMED"

    with pytest.raises(PolicyBenchReviewError, match="never claim GOLD"):
        select_review_candidates([record], record_count=1, seed=1)
