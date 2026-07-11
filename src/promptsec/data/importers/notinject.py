"""Conservative importer for the PIGuard mirror of NotInject."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from promptsec.data.importers.base import (
    BaseImporter,
    ImporterError,
    RawRecord,
    uncertain_annotations,
)

_ORIGINAL_LABEL_FIELDS = ("word_list", "category", "label")
_EXPECTED_TRIGGER_COUNTS = {
    "NotInject_one": 1,
    "NotInject_two": 2,
    "NotInject_three": 3,
}
_REVIEW_REASON = (
    "NotInject trigger-word, category, and benign/detector labels do not determine the "
    "PromptSec-FM instruction, authority, or attack axes; original labels are preserved "
    "for review without SAFE-label migration."
)


class NotInjectImporter(BaseImporter):
    """Preserve NotInject examples without treating benign labels as ground truth."""

    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        text_field, text = self.first_present(raw.payload, self.config.fields.text)
        if not isinstance(text, str):
            raise ImporterError(
                f"NotInject field {text_field!r} must be a string, got {type(text).__name__}"
            )

        expected_trigger_count = _EXPECTED_TRIGGER_COUNTS.get(raw.split)
        word_list = raw.payload.get("word_list")
        if expected_trigger_count is not None and (
            not isinstance(word_list, list) or len(word_list) != expected_trigger_count
        ):
            actual = len(word_list) if isinstance(word_list, list) else type(word_list).__name__
            raise ImporterError(
                f"NotInject split {raw.split!r} expects word_list length "
                f"{expected_trigger_count}, got {actual} at record {raw.index}"
            )

        original_labels = {
            field: raw.payload[field] for field in _ORIGINAL_LABEL_FIELDS if field in raw.payload
        }
        field_mappings = [
            {
                "source": text_field,
                "target": "content.text",
                "method": "COPY_EXACT",
            }
        ]
        field_mappings.extend(
            {
                "source": field,
                "target": (f"metadata.dataset_provenance.source_record.original_labels.{field}"),
                "method": "PRESERVE_EXACT_NO_TAXONOMY_INFERENCE",
            }
            for field in original_labels
        )

        category = raw.payload.get("category")
        content_overrides = {"language": "und"} if category == "Multilingual" else None
        yield self.build_record(
            raw,
            text=text,
            source_text=text,
            annotations=uncertain_annotations(),
            original_labels=original_labels,
            mapping_status="NEEDS_REVIEW",
            field_mappings=field_mappings,
            review_reasons=[_REVIEW_REASON],
            content_overrides=content_overrides,
        )


__all__ = ["NotInjectImporter"]
