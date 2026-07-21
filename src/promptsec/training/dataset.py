"""Integrity-gated official-split loading and tokenized PyTorch datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.baselines.dataset import build_split_audit, load_label_vocabularies
from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import iter_jsonl, read_json_object
from promptsec.policybench.splitting import PUBLISHED_SPLITS
from promptsec.policybench.validation import (
    iter_dataset_records,
    validate_record_collection,
    validate_release_directory,
)
from promptsec.training.labels import (
    HEADS,
    MULTILABEL_HEADS,
    LabelMapping,
    encode_labels,
    load_label_mappings,
)
from promptsec.training.serialization import record_sections
from promptsec.training.token_budget import encode_with_section_budget

EVALUATION_SPLITS = (
    "validation",
    "test_policy_family_ood",
    "test_domain_ood",
    "test_language_ood",
    "test_counterfactual",
)
EXPECTED_SPLIT_COUNTS = {
    "train": 1012,
    "validation": 242,
    "test_policy_family_ood": 284,
    "test_domain_ood": 491,
    "test_language_ood": 3000,
    "test_counterfactual": 344,
    "human_review_candidates": 627,
}


class TrainingDatasetError(ValueError):
    """Raised before model loading when release integrity or leakage checks fail."""


@dataclass(slots=True)
class TrainingDatasetBundle:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    split_hashes: dict[str, str]
    records_by_split: dict[str, list[dict[str, Any]]]
    mappings: dict[str, LabelMapping]
    integrity_report: dict[str, Any]
    split_audit: dict[str, Any]
    annotation_schema: Path


def resolve_annotation_schema(dataset: str | Path) -> Path:
    root = Path(dataset).resolve()
    repository = Path(__file__).resolve().parents[3]
    candidates = (
        root / "schemas" / "promptsec-annotation-v1.schema.json",
        root.parent / "schemas" / "promptsec-annotation-v1.schema.json",
        repository / "schemas" / "promptsec-annotation-v1.schema.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TrainingDatasetError("cannot locate frozen promptsec-annotation-v1 schema")


def _verify_colab_package(root: Path) -> dict[str, Any] | None:
    path = root / "colab_input_manifest.json"
    if not path.is_file():
        return None
    manifest = read_json_object(path)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TrainingDatasetError("Colab input manifest has no files mapping")
    for name, digest in files.items():
        candidate = root / name
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise TrainingDatasetError(f"Colab package checksum mismatch: {name}")
    return manifest


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(value, Mapping):
        raise TrainingDatasetError(f"record {record.get('id')!r}: PolicyBench extension missing")
    return value


def _distribution(records: Sequence[Mapping[str, Any]], getter: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(getter(record)) for record in records).items()))


def _validate_training_coverage(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    mappings: Mapping[str, LabelMapping],
) -> None:
    train_observed: dict[str, set[str]] = {head: set() for head in HEADS}
    for record in records_by_split["train"]:
        for head, mapping in mappings.items():
            parent = "derived" if head == "prompt_injection_verdict" else "annotations"
            value = record[parent][head]
            train_observed[head].update(value if mapping.multilabel else [value])
    for split in EVALUATION_SPLITS:
        for head, mapping in mappings.items():
            observed: set[str] = set()
            for record in records_by_split[split]:
                parent = "derived" if head == "prompt_injection_verdict" else "annotations"
                value = record[parent][head]
                observed.update(value if mapping.multilabel else [value])
            missing = observed.difference(train_observed[head])
            if missing:
                raise TrainingDatasetError(
                    f"blocking split issue: {split}/{head} labels absent from train: "
                    f"{sorted(missing)}"
                )


def load_training_dataset(dataset: str | Path) -> TrainingDatasetBundle:
    """Load only official views after full/local or packaged checksum validation."""

    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    split_report_path = root / "split_report.json"
    if not manifest_path.is_file() or not split_report_path.is_file():
        raise TrainingDatasetError("manifest.json or split_report.json is missing")
    package_manifest = _verify_colab_package(root)
    manifest = read_json_object(manifest_path)
    split_report = read_json_object(split_report_path)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in PUBLISHED_SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise TrainingDatasetError(f"official split is missing: {path.name}")
        records_by_split[split] = iter_jsonl(path)
        if len(records_by_split[split]) != EXPECTED_SPLIT_COUNTS[split]:
            raise TrainingDatasetError(f"unexpected record count for {split}")
    if package_manifest is None:
        canonical = iter_dataset_records(root)
        validation = validate_release_directory(root, canonical)
        if validation["validation_status"] != "PASS":
            raise TrainingDatasetError(f"release validation failed: {validation['errors']}")
    else:
        source_release_validation = package_manifest.get("source_release_validation", {})
        validation = {
            "validation_status": "PASS",
            "source": "colab_input_manifest",
            "files_checked": len(package_manifest["files"]),
            "checksums_checked": source_release_validation.get("checksums_checked"),
            "split_files_checked": source_release_validation.get("split_files_checked"),
        }
    all_records = [record for split in PUBLISHED_SPLITS for record in records_by_split[split]]
    canonical_validation = validate_record_collection(all_records)
    if canonical_validation["validation_status"] != "PASS":
        raise TrainingDatasetError(
            "canonical schema/semantic validation failed for "
            f"{canonical_validation['invalid_records']} records"
        )
    ids = [record.get("id") for record in all_records]
    if len(all_records) != 6000 or len(set(ids)) != 6000:
        raise TrainingDatasetError("official split views are not an exclusive 6,000-ID partition")
    if (
        manifest.get("records") != 6000
        or manifest.get("data_quality") != "SILVER_VALIDATED"
        or manifest.get("human_validation_status") != "PENDING"
        or manifest.get("automatic_gold_records") != 0
    ):
        raise TrainingDatasetError("release manifest violates SILVER/PENDING/no-Gold constraints")
    assignments = split_report.get("assignments", {})
    for split, records in records_by_split.items():
        for record in records:
            extension = _extension(record)
            if assignments.get(record["id"]) != split or extension.get("dataset_split") != split:
                raise TrainingDatasetError(f"record {record['id']!r}: split assignment mismatch")
            if (
                extension.get("data_quality") != "SILVER_VALIDATED"
                or extension.get("human_validation_status") != "PENDING"
                or record["annotations"].get("annotator_confidence") != 0.0
            ):
                raise TrainingDatasetError(f"record {record['id']!r}: SILVER state changed")
    language = _distribution(all_records, lambda record: record["content"]["language"])
    if language != {"en": 3000, "fr": 3000}:
        raise TrainingDatasetError(f"unexpected language distribution: {language}")
    schema_path = resolve_annotation_schema(root)
    mappings = load_label_mappings(schema_path)
    for record in all_records:
        encode_labels(record, mappings)
    _validate_training_coverage(records_by_split, mappings)
    baseline_vocabularies = load_label_vocabularies(schema_path)
    split_audit = build_split_audit(records_by_split, split_report, baseline_vocabularies)
    if split_audit["leakage_detected"]:
        raise TrainingDatasetError(f"official split leakage: {split_audit['leakage_failures']}")
    split_audit["all_head_distributions"] = {
        split: {
            head: _head_distribution(records, head, mappings[head].multilabel) for head in HEADS
        }
        for split, records in records_by_split.items()
    }
    split_hashes = {split: sha256_file(root / f"{split}.jsonl") for split in PUBLISHED_SPLITS}
    integrity = {
        **validation,
        "canonical_validation": canonical_validation,
        "records": 6000,
        "unique_ids": 6000,
        "languages": language,
        "data_quality": "SILVER_VALIDATED",
        "human_validation_status": "PENDING",
        "annotator_confidence": 0.0,
        "automatic_gold_records": 0,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "leakage_detected": False,
    }
    return TrainingDatasetBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        split_hashes=split_hashes,
        records_by_split=records_by_split,
        mappings=mappings,
        integrity_report=integrity,
        split_audit=split_audit,
        annotation_schema=schema_path,
    )


def _head_distribution(
    records: Sequence[Mapping[str, Any]], head: str, multilabel: bool
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    parent = "derived" if head == "prompt_injection_verdict" else "annotations"
    for record in records:
        value = record[parent][head]
        if multilabel:
            counts.update(value)
            if not value:
                counts["[EMPTY]"] += 1
        else:
            counts[value] += 1
    return dict(sorted(counts.items()))


class PolicyBenchTorchDataset:
    """Lazy cached tokenization over canonical records; source objects remain immutable."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        mappings: Mapping[str, LabelMapping],
        *,
        max_length: int,
        head_tail_ratio: float = 0.75,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.mappings = mappings
        self.max_length = max_length
        self.head_tail_ratio = head_tail_ratio
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index not in self._cache:
            record = self.records[index]
            encoded = encode_with_section_budget(
                self.tokenizer,
                record_sections(record),
                max_length=self.max_length,
                head_tail_ratio=self.head_tail_ratio,
            )
            extension = record["extensions"]["policybench_v0_1"]
            counterfactual = extension.get("counterfactual")
            counterfactual = counterfactual if isinstance(counterfactual, Mapping) else {}
            self._cache[index] = {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "labels": encode_labels(record, self.mappings),
                "metadata": {
                    "id": record["id"],
                    "language": record["content"]["language"],
                    "domain": extension["blueprint"]["domain"],
                    "category": extension["blueprint"]["category"],
                    "counterfactual_group_id": counterfactual.get("counterfactual_group_id"),
                    "counterfactual_type": counterfactual.get("counterfactual_type"),
                    "expected_label_changes": list(
                        counterfactual.get("expected_label_changes", [])
                    ),
                },
                "truncation": encoded.statistics,
            }
        return self._cache[index]


class MultitaskCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        padded = self.tokenizer.pad(
            [
                {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
                for item in examples
            ],
            padding=True,
            return_tensors="pt",
        )
        labels = {
            head: torch.tensor(
                [item["labels"][head] for item in examples],
                dtype=torch.float32 if head in MULTILABEL_HEADS else torch.long,
            )
            for head in HEADS
        }
        return {
            **padded,
            "labels": labels,
            "metadata": [item["metadata"] for item in examples],
            "truncation": [item["truncation"] for item in examples],
        }


def summarize_truncation(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"records": 0}
    lengths = sorted(int(item["encoded_length"]) for item in items)

    def percentile(value: float) -> int:
        index = min(len(lengths) - 1, round((len(lengths) - 1) * value))
        return lengths[index]

    return {
        "records": len(items),
        "records_truncated": sum(bool(item["record_truncated"]) for item in items),
        "truncation_rate": sum(bool(item["record_truncated"]) for item in items) / len(items),
        "truncation_rate_by_section": {
            section: sum(bool(item["truncated_sections"][section]) for item in items) / len(items)
            for section in (
                "protected_policy",
                "user_goal",
                "source",
                "available_capabilities",
                "candidate",
            )
        },
        "token_length_percentiles": {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": lengths[-1],
        },
        "unknown_token_statistics": {
            section: {
                "unknown_tokens": sum(
                    int(item.get("unknown_token_counts", {}).get(section) or 0) for item in items
                ),
                "original_tokens": sum(
                    int(item["original_token_lengths"][section]) for item in items
                ),
            }
            for section in (
                "protected_policy",
                "user_goal",
                "source",
                "available_capabilities",
                "candidate",
            )
        },
    }
