from __future__ import annotations

import pytest

from promptsec.data.quality.statistics import compute_statistics, render_statistics_markdown


def _record(
    record_id: str,
    *,
    source: str,
    language: str,
    domain: str,
    delivery_mode: str,
    text: str,
    tier: str,
    confidence: float,
    requires_manual_review: bool,
    attack_families: list[str],
    attack_objectives: list[str],
    spans: list[dict],
    instruction_presence: str = "INSTRUCTION_PRESENT",
    instruction_presentation: str = "OPERATIVE",
    instruction_addressee: str = "MODEL_OR_AGENT",
    user_goal_alignment: str = "MISALIGNED",
    protected_policy_alignment: str = "COMPLIANT",
    authority_status: str = "OUTSIDE_AUTHORITY",
    agentic_source: dict | None = None,
) -> dict:
    record = {
        "id": record_id,
        "content": {
            "text": text,
            "language": language,
            "delivery_mode": delivery_mode,
            "source_role": "USER",
            "content_origin": "CHAT_MESSAGE",
            "ingestion_path": "CHAT_INPUT",
            "modality": "TEXT",
            "source_integrity": "VERIFIED",
        },
        "annotations": {
            "instruction_presence": instruction_presence,
            "instruction_presentation": instruction_presentation,
            "instruction_addressee": instruction_addressee,
            "user_goal_alignment": user_goal_alignment,
            "protected_policy_alignment": protected_policy_alignment,
            "authority_status": authority_status,
            "attack_families": attack_families,
            "attack_objectives": attack_objectives,
            "spans": spans,
            "annotation_status": "CONFIRMED",
            "annotator_confidence": confidence,
        },
        "extensions": {
            "quality_v0_1": {
                "grouping": {"domain": domain},
                "mapping_quality": {
                    "annotation_tier": tier,
                    "mapping_confidence": confidence,
                    "requires_manual_review": requires_manual_review,
                    "review_reasons": [],
                },
            }
        },
        "metadata": {"dataset_provenance": {"source_dataset": {"id": source}}},
    }
    if agentic_source is not None:
        record["extensions"]["agentic_source"] = agentic_source
    return record


def _corpus() -> list[dict]:
    return [
        _record(
            "one",
            source="promptinject",
            language="en",
            domain="direct_prompt",
            delivery_mode="DIRECT",
            text="abc",
            tier="DETERMINISTIC_MAPPING",
            confidence=1.0,
            requires_manual_review=False,
            attack_families=["PROMPT_INJECTION"],
            attack_objectives=["TASK_HIJACKING"],
            spans=[{"start": 0, "end": 3, "type": "INJECTION_PAYLOAD"}],
        ),
        _record(
            "two",
            source="bipia",
            language="en",
            domain="document",
            delivery_mode="INDIRECT",
            text="éé",
            tier="HEURISTIC_MAPPING",
            confidence=0.6,
            requires_manual_review=True,
            attack_families=["PROMPT_INJECTION"],
            attack_objectives=["TASK_HIJACKING", "SENSITIVE_DATA_EXFILTRATION"],
            spans=[
                {"start": 0, "end": 1, "type": "DIRECTIVE"},
                {"start": 0, "end": 2, "type": "AUTHORITY_CLAIM"},
            ],
            protected_policy_alignment="UNDETERMINED",
        ),
        _record(
            "three",
            source="notinject",
            language="und",
            domain="hard_negative",
            delivery_mode="DIRECT",
            text="",
            tier="UNANNOTATED",
            confidence=0.0,
            requires_manual_review=True,
            attack_families=[],
            attack_objectives=[],
            spans=[],
            instruction_presence="UNDETERMINED",
            instruction_presentation="UNKNOWN",
            instruction_addressee="UNKNOWN",
            user_goal_alignment="UNDETERMINED",
            protected_policy_alignment="UNDETERMINED",
            authority_status="UNKNOWN",
        ),
    ]


def test_compute_statistics_covers_distributions_mapping_and_unknown_axes() -> None:
    stats = compute_statistics(_corpus())

    assert stats["total_records"] == 3
    assert stats["distributions"] == {
        "source": {"bipia": 1, "notinject": 1, "promptinject": 1},
        "language": {"en": 2, "und": 1},
        "domain": {"direct_prompt": 1, "document": 1, "hard_negative": 1},
        "attack_family": {"NONE": 1, "PROMPT_INJECTION": 2},
        "attack_objective": {
            "NONE": 1,
            "SENSITIVE_DATA_EXFILTRATION": 1,
            "TASK_HIJACKING": 2,
        },
        "delivery_mode": {"DIRECT": 2, "INDIRECT": 1},
        "annotation_tier": {
            "DETERMINISTIC_MAPPING": 1,
            "HEURISTIC_MAPPING": 1,
            "UNANNOTATED": 1,
        },
        "benchmark_suite": {},
        "user_tool": {},
        "attacker_tool": {},
        "injecagent_attack_category": {},
        "injecagent_setting": {},
        "agentdojo_suite": {},
        "mapping_confidence": {"0.00": 1, "0.60": 1, "1.00": 1},
    }
    mapping = stats["mapping_quality"]
    assert mapping["deterministic_mapping"] == {"count": 1, "rate": 0.333333}
    assert mapping["heuristic_mapping"] == {"count": 1, "rate": 0.333333}
    assert mapping["requires_manual_review"] == {"count": 2, "rate": 0.666667}
    assert mapping["review_candidates"] == {"count": 2, "rate": 0.666667}
    assert mapping["mapping_confidence"] == {
        "count": 3,
        "min": 0.0,
        "mean": 0.533333,
        "median": 0.6,
        "p95": 1.0,
        "max": 1.0,
    }
    unknown = stats["unknown_or_undetermined"]
    assert unknown["annotations.instruction_presence"]["UNDETERMINED"] == 1
    assert unknown["annotations.instruction_presence"]["UNDETERMINED_rate"] == 0.333333
    assert unknown["annotations.instruction_presentation"]["UNKNOWN"] == 1
    assert unknown["annotations.instruction_presentation"]["UNKNOWN_rate"] == 0.333333
    assert unknown["annotations.instruction_addressee"]["UNKNOWN"] == 1
    assert unknown["annotations.user_goal_alignment"]["combined"] == {
        "count": 1,
        "rate": 0.333333,
    }
    assert unknown["annotations.protected_policy_alignment"]["combined"] == {
        "count": 2,
        "rate": 0.666667,
    }
    assert unknown["annotations.authority_status"]["UNKNOWN"] == 1


def test_compute_statistics_measures_unicode_content_and_spans() -> None:
    stats = compute_statistics(_corpus())

    assert stats["content_length"]["characters"] == {
        "count": 3,
        "min": 0,
        "mean": 1.666667,
        "median": 2.0,
        "p95": 3,
        "max": 3,
    }
    assert stats["content_length"]["utf8_bytes"] == {
        "count": 3,
        "min": 0,
        "mean": 2.333333,
        "median": 3.0,
        "p95": 4,
        "max": 4,
    }
    assert stats["spans"] == {
        "total": 3,
        "records_with_spans": {"count": 2, "rate": 0.666667},
        "count_per_record": {
            "count": 3,
            "min": 0,
            "mean": 1.0,
            "median": 1.0,
            "p95": 2,
            "max": 2,
        },
        "length_characters": {
            "count": 3,
            "min": 1,
            "mean": 2.0,
            "median": 2.0,
            "p95": 3,
            "max": 3,
        },
        "by_type": {"AUTHORITY_CLAIM": 1, "DIRECTIVE": 1, "INJECTION_PAYLOAD": 1},
    }


def test_compute_statistics_empty_corpus_is_explicit() -> None:
    stats = compute_statistics([])

    assert stats["total_records"] == 0
    assert stats["distributions"]["source"] == {}
    assert stats["mapping_quality"]["deterministic_mapping"] == {"count": 0, "rate": 0.0}
    assert stats["content_length"]["characters"]["count"] == 0
    assert stats["spans"]["total"] == 0


def test_compute_statistics_treats_missing_quality_as_unannotated_review_candidate() -> None:
    record = _corpus()[0]
    record["extensions"]["quality_v0_1"].pop("mapping_quality")

    stats = compute_statistics([record])

    assert stats["distributions"]["annotation_tier"] == {"UNANNOTATED": 1}
    assert stats["mapping_quality"]["missing_mapping_confidence"] == {
        "count": 1,
        "rate": 1.0,
    }
    assert stats["mapping_quality"]["review_candidates"] == {"count": 1, "rate": 1.0}


def test_review_candidate_rate_applies_mapping_confidence_threshold() -> None:
    record = _corpus()[0]
    quality = record["extensions"]["quality_v0_1"]["mapping_quality"]
    quality["mapping_confidence"] = 0.7
    quality["requires_manual_review"] = False

    without_threshold = compute_statistics([record])
    with_threshold = compute_statistics([record], review_threshold=0.85)

    assert without_threshold["mapping_quality"]["review_candidates"]["count"] == 0
    assert with_threshold["mapping_quality"]["review_candidates"] == {
        "count": 1,
        "rate": 1.0,
    }


def test_compute_statistics_rejects_invalid_span_bounds() -> None:
    corpus = _corpus()
    corpus[0]["annotations"]["spans"] = [{"start": 3, "end": 2, "type": "DIRECTIVE"}]

    with pytest.raises(ValueError, match="invalid span bounds"):
        compute_statistics(corpus)


def test_render_statistics_markdown_is_deterministic_and_complete() -> None:
    stats = compute_statistics(_corpus())

    first = render_statistics_markdown(stats)
    second = render_statistics_markdown(stats)

    assert first == second
    assert first.endswith("\n")
    assert "# PromptSec-Dataset statistics" in first
    assert "| promptinject | 1 | 33.33% |" in first
    assert "| Content UTF-8 bytes | 3 | 0 | 2.333333 | 3.0 | 4 | 4 |" in first
    assert "| annotations.protected_policy_alignment | 0 | 2 | 2 | 66.67% |" in first


def test_compute_statistics_reports_agentic_source_metadata() -> None:
    injecagent = _record(
        "injecagent-one",
        source="injecagent",
        language="en",
        domain="direct_harm",
        delivery_mode="INDIRECT",
        text="perform the injected action",
        tier="HEURISTIC_MAPPING",
        confidence=0.8,
        requires_manual_review=True,
        attack_families=["PROMPT_INJECTION"],
        attack_objectives=["UNAUTHORIZED_ACTION"],
        spans=[],
        agentic_source={
            "attack_category": "Direct Harm",
            "setting": "enhanced",
            "user_tool": "MailSearch",
            "attacker_tools": ["MailSend", "CalendarWrite"],
        },
    )
    agentdojo = _record(
        "agentdojo-one",
        source="agentdojo",
        language="en",
        domain="workspace",
        delivery_mode="INDIRECT",
        text="static injection goal",
        tier="HEURISTIC_MAPPING",
        confidence=0.55,
        requires_manual_review=True,
        attack_families=["PROMPT_INJECTION"],
        attack_objectives=[],
        spans=[],
        user_goal_alignment="UNDETERMINED",
        protected_policy_alignment="UNDETERMINED",
        authority_status="UNKNOWN",
        agentic_source={"suite_id": "workspace"},
    )

    distributions = compute_statistics([injecagent, agentdojo])["distributions"]

    assert distributions["benchmark_suite"] == {"workspace": 1}
    assert distributions["agentdojo_suite"] == {"workspace": 1}
    assert distributions["user_tool"] == {"MailSearch": 1}
    assert distributions["attacker_tool"] == {"CalendarWrite": 1, "MailSend": 1}
    assert distributions["injecagent_attack_category"] == {"Direct Harm": 1}
    assert distributions["injecagent_setting"] == {"enhanced": 1}
    assert distributions["mapping_confidence"] == {"0.55": 1, "0.80": 1}
