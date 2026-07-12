from __future__ import annotations

from pathlib import Path

import pytest

from promptsec.data.release_config import (
    DatasetReleaseConfig,
    ReleaseConfigError,
)

ROOT = Path(__file__).resolve().parents[2]


def test_loads_phase_32_release_configuration() -> None:
    config = DatasetReleaseConfig.load(ROOT / "configs" / "dataset_v0.1.yaml")

    assert config.identity.id == "promptsec-dataset-v0.1"
    assert config.identity.seed == 3201
    assert config.identity.taxonomy_version == "1.0"
    assert config.mapping_quality.review_threshold == 0.85
    assert config.deduplication.semantic_threshold == 0.72
    assert config.splits.held_out_source == "open_prompt_injection"
    assert config.splits.notinject_ratios["test_id"] == 0.75
    assert len(config.source_configs) == 4
    assert all(path.is_file() for path in config.source_configs)


def test_rejects_split_ratios_that_do_not_sum_to_one(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "dataset_v0.1.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(source.replace("test_id: 0.15", "test_id: 0.25", 1), encoding="utf-8")

    with pytest.raises(ReleaseConfigError, match="sum to 1.0"):
        DatasetReleaseConfig.load(invalid)


def test_loads_phase_33_release_with_provisional_agentic_split() -> None:
    config = DatasetReleaseConfig.load(ROOT / "configs" / "dataset_v0.2.yaml")

    assert config.schema_version == "0.2"
    assert config.identity.id == "promptsec-dataset-v0.2"
    assert config.splits.agentic_sources == ("agentdojo", "injecagent")
    assert config.split_names[-1] == "test_agentic_provisional"
    assert len(config.source_configs) == 6
    assert all(path.is_file() for path in config.source_configs)
