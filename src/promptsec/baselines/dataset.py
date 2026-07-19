"""Manifest-driven PolicyBench loading, label extraction, and split auditing."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.baselines.config import EVALUATION_SPLITS, TARGETS
from promptsec.data.hashing import sha256_file
from promptsec.policybench.io import read_json_object, safe_child
from promptsec.policybench.splitting import PUBLISHED_SPLITS
from promptsec.policybench.validation import iter_dataset_records, validate_release_directory

TARGET_PATHS = {
    "prompt_injection_verdict": ("derived", "prompt_injection_verdict"),
    "user_goal_alignment": ("annotations", "user_goal_alignment"),
    "protected_policy_alignment": ("annotations", "protected_policy_alignment"),
    "authority_status": ("annotations", "authority_status"),
    "instruction_presentation": ("annotations", "instruction_presentation"),
}


class BaselineDatasetError(ValueError):
    """Raised when the release cannot safely be used for this benchmark."""


@dataclass(slots=True)
class ReleaseBundle:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    split_hashes: dict[str, str]
    records_by_split: dict[str, list[dict[str, Any]]]
    label_vocabularies: dict[str, tuple[str, ...]]
    split_audit: dict[str, Any]
    release_file_hashes: dict[str, str]


def _schema_vocabulary(schema: Mapping[str, Any], target: str) -> tuple[str, ...]:
    if target == "prompt_injection_verdict":
        value = schema["properties"]["derived"]["properties"][target]["enum"]
    else:
        value = schema["properties"]["annotations"]["properties"][target]["enum"]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BaselineDatasetError(f"canonical schema has no enum for {target}")
    return tuple(value)


def load_label_vocabularies(schema_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read frozen label order directly from annotation schema v1.0."""

    schema = read_json_object(schema_path)
    return {target: _schema_vocabulary(schema, target) for target in TARGETS}


def extract_label(record: Mapping[str, Any], target: str, labels: Sequence[str]) -> str:
    """Extract one canonical truth value and reject vocabulary drift."""

    if target not in TARGET_PATHS:
        raise BaselineDatasetError(f"unknown target: {target}")
    parent, field = TARGET_PATHS[target]
    container = record.get(parent)
    value = container.get(field) if isinstance(container, Mapping) else None
    if value not in labels:
        raise BaselineDatasetError(
            f"record {record.get('id')!r}: {target} value {value!r} is outside frozen vocabulary"
        )
    return value


def validate_checksum_index(dataset: str | Path) -> dict[str, str]:
    """Validate every path in the release checksum index and return expected hashes."""

    root = Path(dataset).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise BaselineDatasetError("release checksums.sha256 is missing")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise BaselineDatasetError("malformed release checksum index")
        digest, name = parts
        path = safe_child(root, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise BaselineDatasetError(f"release checksum mismatch: {name}")
        expected[name] = digest
    return expected


def _release_file_hashes(root: Path) -> dict[str, str]:
    names = list(validate_checksum_index(root))
    names.append("checksums.sha256")
    return {name: sha256_file(root / name) for name in sorted(names)}


def discover_official_split_paths(dataset: str | Path) -> dict[str, Path]:
    """Resolve official split views only when the release checksum index names them."""

    root = Path(dataset).resolve()
    checksum_path = root / "checksums.sha256"
    if not (root / "manifest.json").is_file() or not checksum_path.is_file():
        raise BaselineDatasetError("release manifest or checksum index is missing")
    indexed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise BaselineDatasetError("malformed release checksum index")
        indexed.add(parts[1])
    paths = {split: root / f"{split}.jsonl" for split in PUBLISHED_SPLITS}
    missing = [
        path.name for path in paths.values() if path.name not in indexed or not path.is_file()
    ]
    if missing:
        raise BaselineDatasetError(
            f"official split files are missing or not checksummed: {missing}"
        )
    return paths


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    if not isinstance(value, Mapping):
        raise BaselineDatasetError(f"record {record.get('id')!r}: PolicyBench extension missing")
    return value


def _distribution(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value: Any = record
        for part in path:
            value = value.get(part) if isinstance(value, Mapping) else None
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _value_set(records: Sequence[Mapping[str, Any]], getter: Any) -> set[str]:
    return {value for record in records if (value := getter(record)) is not None}


def _overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    shared = sorted(left.intersection(right))
    return {"count": len(shared), "values": shared}


def build_split_audit(
    records_by_split: Mapping[str, Sequence[dict[str, Any]]],
    split_report: Mapping[str, Any],
    label_vocabularies: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build a text-free audit and independently verify forbidden overlaps."""

    audit: dict[str, Any] = {
        "schema_version": "0.1",
        "official_splits": list(PUBLISHED_SPLITS),
        "splits": {},
    }
    for split in PUBLISHED_SPLITS:
        records = records_by_split[split]
        audit["splits"][split] = {
            "records": len(records),
            "language": _distribution(records, ("content", "language")),
            "domain": _distribution(
                records, ("extensions", "policybench_v0_1", "blueprint", "domain")
            ),
            "category": _distribution(
                records, ("extensions", "policybench_v0_1", "blueprint", "category")
            ),
            "targets": {
                target: dict(
                    sorted(
                        Counter(
                            extract_label(record, target, label_vocabularies[target])
                            for record in records
                        ).items()
                    )
                )
                for target in TARGETS
            },
        }

    def counterfactual(record: Mapping[str, Any]) -> str | None:
        value = _extension(record).get("counterfactual")
        return value.get("counterfactual_group_id") if isinstance(value, Mapping) else None

    def grouping(field: str) -> Any:
        return lambda record: _extension(record)["grouping"].get(field)

    atomic_fields = {
        "counterfactual_group_id": counterfactual,
        "semantic_duplicate_cluster_id": grouping("semantic_duplicate_cluster_id"),
        "split_group_id": grouping("split_group_id"),
    }
    atomic_locations: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in atomic_fields
    }
    for split, records in records_by_split.items():
        for field, getter in atomic_fields.items():
            for value in _value_set(records, getter):
                atomic_locations[field][value].add(split)
    violations = {
        field: {value: sorted(splits) for value, splits in locations.items() if len(splits) > 1}
        for field, locations in atomic_locations.items()
    }

    train = records_by_split["train"]
    comparisons = {
        "policy_family_train_vs_policy_ood": _overlap(
            _value_set(train, grouping("policy_family")),
            _value_set(records_by_split["test_policy_family_ood"], grouping("policy_family")),
        ),
        "domain_train_vs_domain_ood": _overlap(
            _value_set(train, lambda record: _extension(record)["blueprint"].get("domain")),
            _value_set(
                records_by_split["test_domain_ood"],
                lambda record: _extension(record)["blueprint"].get("domain"),
            ),
        ),
        "language_train_vs_language_ood": _overlap(
            _value_set(train, lambda record: record["content"].get("language")),
            _value_set(
                records_by_split["test_language_ood"],
                lambda record: record["content"].get("language"),
            ),
        ),
    }
    raw_family_overlap: dict[str, dict[str, dict[str, Any]]] = {}
    for field in (
        "policy_family",
        "scenario_template_family",
        "attack_template_family",
        "base_generation_family",
    ):
        getter = grouping(field)
        raw_family_overlap[field] = {
            split: _overlap(_value_set(train, getter), _value_set(records_by_split[split], getter))
            for split in EVALUATION_SPLITS
        }

    constraints = split_report.get("constraints", {})
    required_constraints = (
        "all_satisfied",
        "no_transitive_group_leakage",
        "no_counterfactual_group_leakage",
        "no_semantic_duplicate_leakage",
        "held_out_domain_absent_from_train",
        "held_out_language_absent_from_train",
        "policy_family_ood_absent_from_train",
    )
    failed_constraints = [
        name for name in required_constraints if constraints.get(name) is not True
    ]
    forbidden_comparison_overlap = {
        name: value for name, value in comparisons.items() if value["count"]
    }
    atomic_violations = {name: values for name, values in violations.items() if values}
    audit["group_overlap_checks"] = {
        "globally_atomic": violations,
        "held_out_comparisons": comparisons,
        "raw_family_overlap": raw_family_overlap,
        "raw_family_overlap_note": (
            "Policy/scenario/attack/base families are language-namespaced by the frozen splitter; "
            "raw EN/FR family reuse is intentional and is reported, not treated as leakage."
        ),
        "split_report_constraints": dict(constraints),
    }
    audit["leakage_detected"] = bool(
        failed_constraints or forbidden_comparison_overlap or atomic_violations
    )
    audit["leakage_failures"] = {
        "split_constraints": failed_constraints,
        "held_out_overlaps": forbidden_comparison_overlap,
        "atomic_group_overlaps": atomic_violations,
    }
    return audit


def load_release(
    dataset: str | Path,
    annotation_schema: str | Path,
    *,
    max_train_records: int | None = None,
) -> ReleaseBundle:
    """Validate the immutable release, discover official splits, and load records."""

    root = Path(dataset).resolve()
    manifest_path = root / "manifest.json"
    split_report_path = root / "split_report.json"
    if not manifest_path.is_file() or not split_report_path.is_file():
        raise BaselineDatasetError("dataset must be a complete PolicyBench release directory")
    discover_official_split_paths(root)
    before = _release_file_hashes(root)
    manifest = read_json_object(manifest_path)
    split_report = read_json_object(split_report_path)
    records = iter_dataset_records(root)
    validation = validate_release_directory(root, records)
    if validation.get("validation_status") != "PASS":
        raise BaselineDatasetError(f"release validation failed: {validation.get('errors')}")
    if len(records) != 6000 or manifest.get("records") != 6000:
        raise BaselineDatasetError("the v0.1 benchmark requires exactly 6,000 records")
    ids = [record.get("id") for record in records]
    if len(set(ids)) != len(ids):
        raise BaselineDatasetError("duplicate canonical IDs detected")
    if manifest.get("distributions", {}).get("language") != {"en": 3000, "fr": 3000}:
        raise BaselineDatasetError("release is not the expected 3,000 EN / 3,000 FR build")
    if (
        manifest.get("data_quality") != "SILVER_VALIDATED"
        or manifest.get("human_validation_status") != "PENDING"
        or manifest.get("automatic_gold_records") != 0
    ):
        raise BaselineDatasetError("release manifest is not SILVER_VALIDATED/PENDING/no-Gold")
    assignments = split_report.get("assignments")
    if not isinstance(assignments, Mapping):
        raise BaselineDatasetError("split_report.assignments is missing")
    records_by_split = {split: [] for split in PUBLISHED_SPLITS}
    for record in records:
        extension = _extension(record)
        if (
            extension.get("data_quality") != "SILVER_VALIDATED"
            or extension.get("human_validation_status") != "PENDING"
        ):
            raise BaselineDatasetError(
                f"record {record.get('id')!r} is not SILVER_VALIDATED/PENDING"
            )
        split = assignments.get(record["id"])
        if split not in records_by_split:
            raise BaselineDatasetError(f"record {record['id']!r} has unusable split {split!r}")
        records_by_split[split].append(record)
    for split, split_records in records_by_split.items():
        expected = split_report.get("records_by_split", {}).get(split)
        if expected != len(split_records):
            raise BaselineDatasetError(f"split count mismatch for {split}")
    vocabularies = load_label_vocabularies(annotation_schema)
    audit = build_split_audit(records_by_split, split_report, vocabularies)
    if audit["leakage_detected"]:
        raise BaselineDatasetError(f"official split leakage detected: {audit['leakage_failures']}")
    train_labels = {
        target: {
            extract_label(record, target, vocabularies[target])
            for record in records_by_split["train"]
        }
        for target in TARGETS
    }
    for split in EVALUATION_SPLITS:
        for target in TARGETS:
            observed = {
                extract_label(record, target, vocabularies[target])
                for record in records_by_split[split]
            }
            unseen = observed.difference(train_labels[target])
            if unseen:
                raise BaselineDatasetError(
                    f"blocking split issue: {split}/{target} has unseen labels {sorted(unseen)}"
                )
    if max_train_records is not None:
        if max_train_records < 1:
            raise BaselineDatasetError("max_train_records must be positive")
        records_by_split["train"] = records_by_split["train"][:max_train_records]
    after = _release_file_hashes(root)
    if before != after:
        raise BaselineDatasetError("release changed while it was being loaded")
    split_hashes = {split: sha256_file(root / f"{split}.jsonl") for split in PUBLISHED_SPLITS}
    return ReleaseBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        split_hashes=split_hashes,
        records_by_split=records_by_split,
        label_vocabularies=vocabularies,
        split_audit=audit,
        release_file_hashes=before,
    )


def release_file_hashes(dataset: str | Path) -> dict[str, str]:
    """Public byte-identity helper used by gates and tests."""

    return _release_file_hashes(Path(dataset).resolve())
