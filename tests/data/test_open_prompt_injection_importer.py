from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from promptsec.data.config import SourceConfig
from promptsec.data.importers.base import ImporterError
from promptsec.data.importers.open_prompt_injection import OpenPromptInjectionImporter
from promptsec.data.validation import validate_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "data" / "fixtures" / "open_prompt_injection"


def _importer() -> OpenPromptInjectionImporter:
    config = SourceConfig.load(
        REPOSITORY_ROOT / "configs" / "sources" / "open_prompt_injection.toml"
    )
    return OpenPromptInjectionImporter(config, imported_at="2026-07-11T12:00:00Z")


def _zip_fixture(source: Path, destination: Path) -> Path:
    archive_path = destination / "source_archive.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return archive_path


def test_resolves_exact_target_and_all_present_injected_variants(tmp_path: Path) -> None:
    archive = _zip_fixture(FIXTURES / "valid", tmp_path)
    records = list(_importer().records({"source_archive": archive}))

    assert len(records) == 4
    assert len({record["id"] for record in records}) == 4
    expected_variants = ["base", "short", "med_long", "long"]
    expected_names = [
        "replacement",
        "replacement_short",
        "replacement_med_long",
        "replacement_long",
    ]
    assert [
        record["metadata"]["dataset_provenance"]["source_record"]["original_fields"]["variant"]
        for record in records
    ] == expected_variants
    assert [
        record["metadata"]["dataset_provenance"]["source_record"]["original_fields"][
            "template_name"
        ]
        for record in records
    ] == expected_names

    fixture_root = FIXTURES / "valid" / "Open-Prompt-Injection-fixture"
    expected_target = (fixture_root / "data" / "system_prompts" / "analysis.txt").read_text(
        encoding="utf-8"
    )
    for record, template_name, variant in zip(
        records, expected_names, expected_variants, strict=True
    ):
        expected_injected = (
            fixture_root / "data" / "system_prompts" / f"{template_name}.txt"
        ).read_text(encoding="utf-8")
        provenance = record["metadata"]["dataset_provenance"]
        original_fields = provenance["source_record"]["original_fields"]
        original_labels = provenance["source_record"]["original_labels"]

        assert record["content"]["text"] == expected_injected
        assert record["context"]["user_goal"] == expected_target
        assert original_fields["target_instruction"] == expected_target
        assert original_fields["injected_instruction"] == expected_injected
        assert original_fields["task_config"] == "configs/task_configs/alpha_config.json"
        assert original_fields["task"] == "sentiment_analysis"
        assert original_fields["dataset"] == "fixture_reviews"
        assert original_fields["target_instruction_name"] == "analysis"
        assert original_fields["injected_instruction_name"] == "replacement"
        assert original_fields["variant"] == variant
        assert original_fields["template_name"] == template_name
        assert original_labels == {
            field: original_fields[field]
            for field in (
                "task",
                "task_type",
                "dataset",
                "target_instruction_name",
                "injected_instruction_name",
                "variant",
                "template_name",
            )
        }
        assert provenance["source_record"]["id"].endswith(f"::{template_name}")
        assert provenance["mapping"]["status"] == "NEEDS_REVIEW"
        assert any(
            "downstream dataset examples are not redistributed" in reason
            for reason in provenance["mapping"]["review_reasons"]
        )
        assert validate_record(record) == []


def test_maps_explicit_task_replacement_and_full_exact_span(tmp_path: Path) -> None:
    archive = _zip_fixture(FIXTURES / "valid", tmp_path)
    records = list(_importer().records({"source_archive": archive}))

    for record in records:
        text = record["content"]["text"]
        annotations = record["annotations"]
        assert record["content"]["delivery_mode"] == "INDIRECT"
        assert record["content"]["source_role"] == "EXTERNAL_CONTENT"
        assert annotations["instruction_presence"] == "INSTRUCTION_PRESENT"
        assert annotations["instruction_presentation"] == "OPERATIVE"
        assert annotations["instruction_addressee"] == "MODEL_OR_AGENT"
        assert annotations["user_goal_alignment"] == "MISALIGNED"
        assert annotations["protected_policy_alignment"] == "UNDETERMINED"
        assert annotations["authority_status"] == "OUTSIDE_AUTHORITY"
        assert annotations["attack_families"] == ["PROMPT_INJECTION"]
        assert annotations["attack_objectives"] == ["TASK_HIJACKING"]
        assert annotations["annotation_status"] == "CONFIRMED"
        assert annotations["spans"] == [{"start": 0, "end": len(text), "type": "INJECTION_PAYLOAD"}]
        assert record["derived"]["prompt_injection_verdict"] == "DETECTED"

    # The empty-pair config and target-side _cot template are deliberately not records.
    template_names = {
        record["metadata"]["dataset_provenance"]["source_record"]["original_fields"][
            "template_name"
        ]
        for record in records
    }
    assert "analysis_cot" not in template_names


def test_refuses_missing_configured_template(tmp_path: Path) -> None:
    archive = _zip_fixture(FIXTURES / "missing_reference", tmp_path)

    with pytest.raises(ImporterError, match="referenced template is missing"):
        list(_importer().records({"source_archive": archive}))
