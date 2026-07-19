from __future__ import annotations

from pathlib import Path

import pytest

from promptsec.baselines.dataset import release_file_hashes
from promptsec.training.dataset import EXPECTED_SPLIT_COUNTS, load_training_dataset


@pytest.mark.integration
def test_immutable_policybench_release_loads_only_official_splits() -> None:
    root = Path("data/generated/policybench-codex-v0.1")
    if not root.is_dir():
        pytest.skip("local immutable PolicyBench release is not present")
    before = release_file_hashes(root)
    bundle = load_training_dataset(root)
    after = release_file_hashes(root)
    assert before == after
    assert {name: len(records) for name, records in bundle.records_by_split.items()} == (
        EXPECTED_SPLIT_COUNTS
    )
    assert bundle.integrity_report["leakage_detected"] is False
    assert bundle.integrity_report["automatic_gold_records"] == 0
    assert bundle.integrity_report["data_quality"] == "SILVER_VALIDATED"
    assert bundle.integrity_report["human_validation_status"] == "PENDING"
    assert bundle.integrity_report["annotator_confidence"] == 0.0
