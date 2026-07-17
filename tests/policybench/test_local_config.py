from pathlib import Path

from promptsec.policybench.config import PolicyBenchConfig

ROOT = Path(__file__).resolve().parents[2]


def test_local_config_reuses_plan_without_requiring_api_key() -> None:
    config = PolicyBenchConfig.load(
        ROOT / "configs" / "policybench_local_v0.1.yaml",
        environ={"PROMPTSEC_GENERATION_MODEL": "local-fixture"},
    )
    assert config.target_records == 6000
    assert config.generation.provider == "openai_compatible"
    assert config.generation.model == "local-fixture"
    assert config.generation.base_url == "http://127.0.0.1:11434/v1"
    assert config.generation.authentication == "optional_for_loopback"
    assert config.generation.response_mode == "json_object"
    assert config.generation.concurrency == 1
    assert config.category_counts
    assert config.counterfactual_counts


def test_codex_config_requires_only_an_explicit_model_and_uses_safe_batch_defaults() -> None:
    config = PolicyBenchConfig.load(
        ROOT / "configs" / "policybench_codex_v0.1.yaml",
        environ={"PROMPTSEC_CODEX_MODEL": "gpt-test-codex"},
    )
    assert config.target_records == 6000
    assert config.generation.provider == "codex_cli"
    assert config.generation.model == "gpt-test-codex"
    assert config.generation.records_per_batch == 10
    assert config.generation.concurrency == 1
    assert config.generation.max_retries == 4
    assert config.generation.timeout_seconds == 900
    assert config.generation.ephemeral is True
    assert config.generation.output_schema == "strict"
