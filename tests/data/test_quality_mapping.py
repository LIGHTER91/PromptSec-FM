from __future__ import annotations

import pytest

from promptsec.data.quality.mapping import assess_mapping, build_review_queue


def _record(
    record_id: str,
    *,
    source: str = "fixture",
    mapping_status: str = "DETERMINISTIC",
    authority_status: str = "WITHIN_AUTHORITY",
    user_goal_alignment: str = "ALIGNED",
    protected_policy_alignment: str = "COMPLIANT",
    mapping_quality: dict | None = None,
    quality_extra: dict | None = None,
    text: str = "Review this record.",
) -> dict:
    quality = dict(quality_extra or {})
    if mapping_quality is not None:
        quality["mapping_quality"] = mapping_quality
    return {
        "id": record_id,
        "content": {"text": text},
        "annotations": {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": user_goal_alignment,
            "protected_policy_alignment": protected_policy_alignment,
            "authority_status": authority_status,
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": ["TASK_HIJACKING"],
            "spans": [],
            "annotation_status": "CONFIRMED",
            "annotator_confidence": 1.0,
        },
        "extensions": {"quality_v0_1": quality},
        "metadata": {
            "dataset_provenance": {
                "source_dataset": {"id": source, "name": source},
                "source_record": {"id": f"upstream-{record_id}"},
                "mapping": {
                    "status": mapping_status,
                    "unmapped_labels": [],
                    "review_reasons": [],
                },
            }
        },
    }


def test_assess_mapping_uses_explicit_source_profile() -> None:
    profiles = {
        "fixture": {
            "annotation_tier": "DETERMINISTIC_MAPPING",
            "mapping_confidence": 0.98,
            "requires_manual_review": False,
            "review_reasons": [],
        }
    }

    assert assess_mapping(_record("one"), profiles) == {
        "annotation_tier": "DETERMINISTIC_MAPPING",
        "mapping_confidence": 0.98,
        "requires_manual_review": False,
        "review_reasons": [],
    }


def test_assess_mapping_never_infers_quality_without_a_profile() -> None:
    result = assess_mapping(_record("one", mapping_status="DETERMINISTIC"), {})

    assert result == {
        "annotation_tier": "UNANNOTATED",
        "mapping_confidence": 0.0,
        "requires_manual_review": True,
        "review_reasons": ["NO_SOURCE_MAPPING_PROFILE"],
    }


def test_assess_mapping_provenance_uncertainty_forces_review() -> None:
    record = _record("one", mapping_status="NEEDS_REVIEW")
    provenance_mapping = record["metadata"]["dataset_provenance"]["mapping"]
    provenance_mapping["unmapped_labels"] = ["legacy-category"]
    provenance_mapping["review_reasons"] = ["Source semantics are incomplete."]
    profiles = {
        "fixture": {
            "annotation_tier": "HEURISTIC_MAPPING",
            "mapping_confidence": 0.7,
            "requires_manual_review": False,
            "review_reasons": ["Rule-derived objective."],
        }
    }

    result = assess_mapping(record, profiles)

    assert result["annotation_tier"] == "HEURISTIC_MAPPING"
    assert result["mapping_confidence"] == 0.7
    assert result["requires_manual_review"] is True
    assert result["review_reasons"] == [
        "Rule-derived objective.",
        "PROVENANCE_MAPPING_STATUS_NEEDS_REVIEW",
        "UNMAPPED_SOURCE_LABELS",
        "Source semantics are incomplete.",
    ]


@pytest.mark.parametrize(
    ("profile_update", "error"),
    [
        ({"annotation_tier": "HUMANISH"}, "annotation_tier"),
        ({"mapping_confidence": 1.2}, "mapping_confidence"),
        ({"mapping_confidence": True}, "mapping_confidence"),
        ({"requires_manual_review": "yes"}, "requires_manual_review"),
        ({"review_reasons": "reason"}, "review_reasons"),
    ],
)
def test_assess_mapping_rejects_invalid_profiles(profile_update: dict, error: str) -> None:
    profile = {
        "annotation_tier": "DETERMINISTIC_MAPPING",
        "mapping_confidence": 1.0,
        "requires_manual_review": False,
        "review_reasons": [],
    }
    profile.update(profile_update)

    with pytest.raises((TypeError, ValueError), match=error):
        assess_mapping(_record("one"), {"fixture": profile})


def test_build_review_queue_covers_all_conditions_and_is_deterministic() -> None:
    accepted = _record(
        "accepted",
        source="a-source",
        mapping_quality={
            "annotation_tier": "GOLD_SOURCE",
            "mapping_confidence": 0.8,
            "requires_manual_review": False,
            "review_reasons": [],
        },
    )
    uncertain = _record(
        "uncertain",
        source="z-source",
        authority_status="UNKNOWN",
        user_goal_alignment="UNDETERMINED",
        protected_policy_alignment="UNDETERMINED",
        mapping_quality={
            "annotation_tier": "HEURISTIC_MAPPING",
            "mapping_confidence": 0.79,
            "requires_manual_review": True,
            "review_reasons": ["Needs adjudication."],
        },
        quality_extra={
            "deduplication": {
                "raw_hash": "a" * 64,
                "normalized_hash": "b" * 64,
                "contextual_hash": "c" * 64,
                "semantic_cluster_id": "semantic-1",
            }
        },
        text="x" * 300,
    )
    manual = _record(
        "manual",
        source="b-source",
        mapping_quality={
            "annotation_tier": "DETERMINISTIC_MAPPING",
            "mapping_confidence": 0.95,
            "requires_manual_review": True,
            "review_reasons": [],
        },
        quality_extra={"hashes": {"raw_hash": "d" * 64}},
    )

    queue = build_review_queue([uncertain, accepted, manual], threshold=0.8)

    assert [entry["id"] for entry in queue] == ["manual", "uncertain"]
    assert queue[0]["conditions"] == ["REQUIRES_MANUAL_REVIEW"]
    assert queue[0]["hashes"] == {"raw_hash": "d" * 64}
    assert queue[1]["conditions"] == [
        "MAPPING_CONFIDENCE_BELOW_THRESHOLD",
        "REQUIRES_MANUAL_REVIEW",
        "AUTHORITY_STATUS_UNKNOWN",
        "USER_GOAL_ALIGNMENT_UNDETERMINED",
        "PROTECTED_POLICY_ALIGNMENT_UNDETERMINED",
    ]
    assert queue[1]["source"] == "z-source"
    assert queue[1]["annotations"] == uncertain["annotations"]
    assert queue[1]["mapping_quality"]["annotation_tier"] == "HEURISTIC_MAPPING"
    assert queue[1]["hashes"] == {
        "raw_hash": "a" * 64,
        "normalized_hash": "b" * 64,
        "contextual_hash": "c" * 64,
    }
    assert queue[1]["content_preview"] == "x" * 240
    assert queue[1]["content"]["text"] == "x" * 300
    assert queue[1]["dataset_provenance"] == uncertain["metadata"]["dataset_provenance"]


def test_build_review_queue_treats_missing_confidence_conservatively() -> None:
    record = _record("missing", mapping_quality={"requires_manual_review": False})

    queue = build_review_queue([record], threshold=0.8)

    assert queue[0]["conditions"] == ["MAPPING_CONFIDENCE_MISSING"]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, True])
def test_build_review_queue_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="threshold"):
        build_review_queue([], threshold=threshold)  # type: ignore[arg-type]
