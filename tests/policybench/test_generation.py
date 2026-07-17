from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from promptsec.data.hashing import canonical_json_bytes, sha256_bytes, sha256_json
from promptsec.policybench import generation, validation
from promptsec.policybench.blueprints import (
    BlueprintPlan,
    PolicyDescriptor,
    build_blueprint_plan,
    make_blueprint,
    policy_descriptors_from_catalogues,
)
from promptsec.policybench.config import DOMAIN_ORDER, PolicyBenchConfig
from promptsec.policybench.generation import (
    GenerationOptions,
    PolicyBenchGenerationError,
    generate_policybench,
)
from promptsec.policybench.policies import load_policy_catalogs
from promptsec.policybench.prompts import PromptBundle
from promptsec.policybench.providers import (
    GenerationProviderError,
    GenerationRequest,
    GenerationResponse,
    MockGenerationProvider,
)
from promptsec.policybench.validation import iter_dataset_records, validate_release_directory

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "policybench_v0.1.yaml"
CODEX_CONFIG_PATH = ROOT / "configs" / "policybench_codex_v0.1.yaml"
PROMPT_ROOT = ROOT / "prompts" / "policybench"
POLICY_ROOT = ROOT / "data" / "policybench" / "policies"


@pytest.fixture(scope="module")
def authored_inputs() -> tuple[dict[str, dict[str, Any]], PromptBundle]:
    return load_policy_catalogs(POLICY_ROOT), PromptBundle.load(PROMPT_ROOT)


def _policy_mapping(catalogues: dict[str, dict[str, Any]], policy_id: str) -> dict[str, Any]:
    for catalogue in catalogues.values():
        for policy in catalogue["policies"]:
            if policy["policy_id"] == policy_id:
                return copy.deepcopy(policy)
    raise AssertionError(f"test policy is absent from catalogues: {policy_id}")


@dataclass(frozen=True)
class _TinyEnvironment:
    root: Path
    plan: BlueprintPlan
    bundle: PromptBundle
    catalogues: dict[str, dict[str, Any]]


@pytest.fixture
def tiny_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authored_inputs: tuple[dict[str, dict[str, Any]], PromptBundle],
) -> _TinyEnvironment:
    catalogues, bundle = authored_inputs
    descriptors = policy_descriptors_from_catalogues(catalogues)
    banking = next(item for item in descriptors if item.domain == "banking")
    email = next(item for item in descriptors if item.domain == "email")
    plan = BlueprintPlan(
        seed=20260715,
        target_records=2,
        blueprints=(
            make_blueprint(
                banking,
                category="NO_INSTRUCTION",
                language="en",
                scenario_id="pb_generation_tiny_en",
                seed=20260715,
            ),
            make_blueprint(
                email,
                category="NO_INSTRUCTION",
                language="fr",
                scenario_id="pb_generation_tiny_fr",
                seed=20260715,
            ),
        ),
    )

    monkeypatch.setattr(generation, "_repository_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        generation,
        "load_policy_catalogs",
        lambda _path: copy.deepcopy(catalogues),
    )
    monkeypatch.setattr(
        generation.PromptBundle,
        "load",
        classmethod(lambda _cls, _root=None: bundle),
    )
    monkeypatch.setattr(generation, "build_blueprint_plan", lambda *_args: plan)
    monkeypatch.setattr(
        generation,
        "_counterfactual_plan",
        lambda base_plan, _config, _policies: (base_plan, ()),
    )
    return _TinyEnvironment(
        root=tmp_path,
        plan=plan,
        bundle=bundle,
        catalogues=copy.deepcopy(catalogues),
    )


def _tiny_options(**changes: Any) -> GenerationOptions:
    values: dict[str, Any] = {
        "provider": "mock",
        "model": "deterministic-template-v1",
        "max_records": 2,
        "concurrency": 1,
        "temperature": 0.0,
    }
    values.update(changes)
    return GenerationOptions(**values)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dry_run_is_write_free_and_cannot_claim_gold(
    tiny_environment: _TinyEnvironment,
) -> None:
    result = generate_policybench(
        CONFIG_PATH,
        output=tiny_environment.root / "release",
        options=_tiny_options(dry_run=True),
    )

    assert result.phase_state == "PLAN_VALIDATED"
    assert result.records == 2
    assert result.output is None
    assert result.automatic_gold_records == 0
    assert result.files == ()
    assert list(tiny_environment.root.rglob("*")) == []


def test_codex_prompt_v9_mechanically_covers_observed_semantic_boundaries(
    authored_inputs: tuple[dict[str, dict[str, Any]], PromptBundle],
) -> None:
    _, bundle = authored_inputs
    prompt = bundle.codex_batch
    normalized = " ".join(prompt.text.split())

    assert prompt.prompt_version == "codex_batch_v9"
    for requirement in (
        "amount_eur",
        "output_requirement",
        "external system",
        "système externe",
        "hypothetical scenario",
        "scénario hypothétique",
        "system override",
        "instruction développeur",
        "BROWSE_WEB",
        "MEMORY",
        "omit every reference to confirmation",
        "TRANSACTION requires",
        "DOWNLOAD_ATTACHMENT requires",
    ):
        assert requirement in normalized
    for exact_authority_phrase in (
        "system override",
        "developer instruction",
        "instruction développeur",
        "commande administrateur",
    ):
        assert validation._AUTHORITY_MARKER.fullmatch(exact_authority_phrase)


def test_codex_provider_generates_one_atomic_bilingual_batch_without_fallback(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMPTSEC_CODEX_MODEL", "gpt-test-codex")
    provider = _BatchSequenceProvider()
    output = tiny_environment.root / "codex-release"
    result = generate_policybench(
        CODEX_CONFIG_PATH,
        output=output,
        options=GenerationOptions(max_records=2, concurrency=1),
        generation_provider=provider,
    )

    assert result.records == 2
    assert result.automatic_gold_records == 0
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 2
    records = iter_dataset_records(output)
    assert {record["content"]["language"] for record in records} == {"en", "fr"}
    assert all(
        record["extensions"]["policybench_v0_1"]["data_quality"] == "SILVER_VALIDATED"
        for record in records
    )
    assert all(
        record["extensions"]["policybench_v0_1"]["human_validation_status"] == "PENDING"
        for record in records
    )
    assert all(record["annotations"]["annotator_confidence"] == 0.0 for record in records)


def test_codex_batch_retry_is_atomic_and_bounded(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMPTSEC_CODEX_MODEL", "gpt-test-codex")
    provider = _BatchSequenceProvider(["semantic_error", "valid"])
    output = tiny_environment.root / "codex-retry-release"
    generate_policybench(
        CODEX_CONFIG_PATH,
        output=output,
        options=GenerationOptions(max_records=2, concurrency=1, max_retries=1),
        generation_provider=provider,
    )

    assert len(provider.calls) == 2
    assert provider.calls[0] == provider.calls[1]
    records = iter_dataset_records(output)
    assert all(
        record["extensions"]["policybench_v0_1"]["generation"]["generation_attempt"] == 2
        for record in records
    )


def test_codex_usage_limit_stops_queued_batches_without_semantic_retries(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMPTSEC_CODEX_MODEL", "gpt-test-codex")
    provider = _BatchSequenceProvider(["usage_limit"])

    with pytest.raises(PolicyBenchGenerationError, match="usage limit"):
        generate_policybench(
            CODEX_CONFIG_PATH,
            output=tiny_environment.root / "codex-usage-limit",
            options=GenerationOptions(max_records=12, concurrency=1, max_retries=4),
            generation_provider=provider,
        )

    assert len(provider.calls) == 1
    raw_root = tiny_environment.root / "data" / "generated" / "raw" / "policybench-codex-v0.1"
    attempts = [json.loads(path.read_text("utf-8")) for path in raw_root.rglob("*.json")]
    assert attempts
    assert all(item["status"] == "REJECTED" for item in attempts)
    assert all(
        any("usage limit" in reason.casefold() for reason in item["rejection_reasons"])
        for item in attempts
    )


def test_codex_resume_reuses_partial_artifacts_and_only_generates_missing_records(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMPTSEC_CODEX_MODEL", "gpt-test-codex")
    provider = _BatchSequenceProvider()
    generate_policybench(
        CODEX_CONFIG_PATH,
        output=tiny_environment.root / "codex-first",
        options=GenerationOptions(max_records=2, concurrency=1),
        generation_provider=provider,
    )
    artifact_root = (
        tiny_environment.root / "data" / "generated" / "accepted" / "policybench-codex-v0.1"
    )
    next(iter(sorted(artifact_root.glob("*.json")))).unlink()
    second = generate_policybench(
        CODEX_CONFIG_PATH,
        output=tiny_environment.root / "codex-second",
        options=GenerationOptions(max_records=2, concurrency=1, resume=True),
        generation_provider=provider,
    )

    assert [len(call) for call in provider.calls] == [2, 1]
    assert second.records == 2
    assert second.reused_accepted_artifacts == 1


def test_codex_provider_without_batch_capability_is_rejected_without_fallback(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMPTSEC_CODEX_MODEL", "gpt-test-codex")

    class SingleOnly:
        provider_name = "codex_cli"
        model = "gpt-test-codex"
        model_revision = "codex-cli test"

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return MockGenerationProvider().generate(request)

    with pytest.raises(PolicyBenchGenerationError, match="batch-capable|batch provider"):
        generate_policybench(
            CODEX_CONFIG_PATH,
            output=tiny_environment.root / "codex-invalid-provider",
            options=GenerationOptions(max_records=2, concurrency=1),
            generation_provider=SingleOnly(),
        )


def test_installed_share_assets_support_prompt_validated_dry_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    share_root = tmp_path / "prefix" / "share" / "promptsec-dataset"
    installed_config = share_root / "configs" / CONFIG_PATH.name
    installed_policies = share_root / "policies" / "policybench"
    installed_prompts = share_root / "prompts" / "policybench"
    installed_config.parent.mkdir(parents=True)
    installed_policies.mkdir(parents=True)
    installed_prompts.mkdir(parents=True)
    shutil.copy2(CONFIG_PATH, installed_config)
    for source in POLICY_ROOT.glob("*.yaml"):
        shutil.copy2(source, installed_policies / source.name)
    for source in PROMPT_ROOT.glob("*.txt"):
        shutil.copy2(source, installed_prompts / source.name)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        generation,
        "_repository_root",
        lambda _path: pytest.fail("installed configs must not fall back to checkout discovery"),
    )

    observed_policy_roots: list[Path] = []
    observed_prompt_roots: list[Path] = []
    real_policy_loader = generation.load_policy_catalogs
    real_prompt_loader = generation.PromptBundle.load.__func__

    def load_policies(path: str | Path) -> dict[str, dict[str, Any]]:
        observed_policy_roots.append(Path(path).resolve())
        return real_policy_loader(path)

    def load_prompts(cls: type[PromptBundle], root: str | Path | None = None) -> PromptBundle:
        assert root is not None
        observed_prompt_roots.append(Path(root).resolve())
        return real_prompt_loader(cls, root)

    monkeypatch.setattr(generation, "load_policy_catalogs", load_policies)
    monkeypatch.setattr(generation.PromptBundle, "load", classmethod(load_prompts))

    result = generate_policybench(
        installed_config,
        options=GenerationOptions(dry_run=True, max_records=12),
    )

    assert result.phase_state == "PLAN_VALIDATED"
    assert result.records == 12
    assert result.output is None
    assert observed_policy_roots == [installed_policies.resolve()]
    assert observed_prompt_roots == [installed_prompts.resolve()]
    assert list(workspace.iterdir()) == []


class _SequenceProvider:
    provider_name = "test_sequence"
    model = "deterministic-template-v1"
    model_revision = "test"

    def __init__(self, steps: list[str]) -> None:
        self.steps = steps
        self.calls = 0
        self._valid = MockGenerationProvider(
            model=self.model,
            model_revision=self.model_revision,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        step = self.steps[self.calls]
        self.calls += 1
        if step == "provider_error":
            raise GenerationProviderError("provider returned malformed JSON")
        if step == "provider_binary_error":
            raw = b"\xffprovider rejection"
            raise GenerationProviderError(
                "provider returned invalid bytes",
                code="INVALID_UTF8",
                raw_sha256=sha256_bytes(raw),
            )
        response = replace(
            self._valid.generate(request),
            provider=self.provider_name,
            model_revision=self.model_revision,
        )
        if step == "semantic_error":
            data = {**response.data, "scenario_id": "pb_wrong_scenario"}
            return replace(
                response,
                data=data,
                raw_text=canonical_json_bytes(data).decode("utf-8"),
            )
        assert step == "valid"
        return response


class _BatchSequenceProvider:
    provider_name = "codex_cli"
    model = "gpt-test-codex"
    model_revision = "codex-cli test"

    def __init__(self, steps: list[str] | None = None) -> None:
        self.steps = list(steps or [])
        self.calls: list[tuple[str, ...]] = []
        self._mock = MockGenerationProvider()

    def generate(self, _request: GenerationRequest) -> GenerationResponse:
        raise AssertionError("codex_cli must never use one-process-per-record generation")

    def generate_batch(
        self,
        requests: Any,
        *,
        instructions: str,
    ) -> tuple[GenerationResponse, ...]:
        assert "inert" in instructions.casefold()
        materialized = tuple(requests)
        self.calls.append(tuple(request.request_id for request in materialized))
        step = self.steps.pop(0) if self.steps else "valid"
        if step == "usage_limit":
            raise GenerationProviderError(
                "Codex account usage limit reached",
                code="USAGE_LIMIT",
            )
        responses = []
        for index, request in enumerate(materialized):
            response = replace(
                self._mock.generate(request),
                provider=self.provider_name,
                model=self.model,
                model_revision=self.model_revision,
            )
            if step == "semantic_error" and index == 0:
                data = {**response.data, "language": "de"}
                response = replace(
                    response,
                    data=data,
                    raw_text=canonical_json_bytes(data).decode("utf-8"),
                )
            responses.append(response)
        return tuple(responses)


def _retry_blueprint_and_policy(
    catalogues: dict[str, dict[str, Any]],
    scenario_id: str,
) -> tuple[Any, dict[str, Any]]:
    descriptor = next(
        item for item in policy_descriptors_from_catalogues(catalogues) if item.domain == "banking"
    )
    blueprint = make_blueprint(
        descriptor,
        category="ALIGNED_AND_COMPLIANT",
        language="en",
        scenario_id=scenario_id,
        seed=71,
    )
    return blueprint, _policy_mapping(catalogues, descriptor.policy_id)


@pytest.mark.parametrize(
    ("first_step", "expected_reason"),
    [
        ("semantic_error", "SCENARIO_ID_MISMATCH"),
        ("provider_error", "STRUCTURE_VALIDATION_FAILED"),
    ],
)
def test_generation_retries_semantic_and_provider_failures_then_accepts(
    tmp_path: Path,
    authored_inputs: tuple[dict[str, dict[str, Any]], PromptBundle],
    first_step: str,
    expected_reason: str,
) -> None:
    catalogues, bundle = authored_inputs
    blueprint, policy = _retry_blueprint_and_policy(catalogues, f"pb_retry_{first_step}")
    provider = _SequenceProvider([first_step, "valid"])

    accepted = generation._acquire_one(
        blueprint,
        policy,
        provider,
        bundle,
        raw_root=tmp_path / "raw",
        generated_at="2026-07-15T00:00:00Z",
        temperature=0.0,
        maximum_candidate_characters=12_000,
        maximum_failed_log_characters=4_096,
        max_retries=1,
    )

    assert provider.calls == 2
    assert accepted.metadata.generation_attempt == 2
    assert len(accepted.metadata.failed_attempts) == 1
    assert expected_reason in repr(accepted.metadata.failed_attempts[0])
    assert "GOLD" not in repr(accepted)
    attempts = sorted((tmp_path / "raw" / blueprint.scenario_id).glob("*.json"))
    assert [json.loads(path.read_text("utf-8"))["status"] for path in attempts] == [
        "REJECTED",
        "ACCEPTED_PENDING_CORPUS_VALIDATION",
    ]


def test_generation_preserves_byte_hash_for_undecodable_provider_failure(
    tmp_path: Path,
    authored_inputs: tuple[dict[str, dict[str, Any]], PromptBundle],
) -> None:
    catalogues, bundle = authored_inputs
    blueprint, policy = _retry_blueprint_and_policy(catalogues, "pb_retry_binary_error")
    provider = _SequenceProvider(["provider_binary_error", "valid"])

    accepted = generation._acquire_one(
        blueprint,
        policy,
        provider,
        bundle,
        raw_root=tmp_path / "raw",
        generated_at="2026-07-15T00:00:00Z",
        temperature=0.0,
        maximum_candidate_characters=12_000,
        maximum_failed_log_characters=4_096,
        max_retries=1,
    )

    expected_hash = sha256_bytes(b"\xffprovider rejection")
    assert accepted.metadata.failed_attempts[0]["raw_generation_sha256"] == expected_hash
    raw_attempt = json.loads(
        (tmp_path / "raw" / blueprint.scenario_id / "attempt_001.json").read_text("utf-8")
    )
    assert raw_attempt["unparsed_response"] == {
        "bounded_prefix_only": False,
        "raw_generation_sha256": expected_hash,
        "raw_text": None,
    }


def test_retry_exhaustion_remains_bounded(
    tmp_path: Path,
    authored_inputs: tuple[dict[str, dict[str, Any]], PromptBundle],
) -> None:
    catalogues, bundle = authored_inputs
    blueprint, policy = _retry_blueprint_and_policy(catalogues, "pb_retry_exhausted")
    provider = _SequenceProvider(["provider_error", "provider_error"])
    with pytest.raises(PolicyBenchGenerationError, match="exhausted 2 attempts"):
        generation._acquire_one(
            blueprint,
            policy,
            provider,
            bundle,
            raw_root=tmp_path / "raw",
            generated_at="2026-07-15T00:00:00Z",
            temperature=0.0,
            maximum_candidate_characters=12_000,
            maximum_failed_log_characters=4_096,
            max_retries=1,
        )
    assert provider.calls == 2


def test_acquisition_fingerprint_binds_endpoint_and_model() -> None:
    common = {
        "config_sha256": "a" * 64,
        "effective_seed": 20260715,
        "provider": "openai_compatible",
        "model_revision": None,
        "authentication": "optional_for_loopback",
        "response_mode": "json_object",
        "temperature": 0.7,
        "prompt_hashes": {"system": "b" * 64},
        "policy_catalogue_hashes": {"banking": "c" * 64},
    }
    original = generation._acquisition_fingerprint(
        **common,
        model="model-a",
        base_url="http://127.0.0.1:11434/v1",
    )
    assert original != generation._acquisition_fingerprint(
        **common,
        model="model-b",
        base_url="http://127.0.0.1:11434/v1",
    )
    assert original != generation._acquisition_fingerprint(
        **common,
        model="model-a",
        base_url="http://127.0.0.1:1234/v1",
    )


def test_codex_fingerprint_binds_cli_model_batch_reasoning_schema_and_taxonomy() -> None:
    common = {
        "config_sha256": "a" * 64,
        "effective_seed": 20260715,
        "provider": "codex_cli",
        "model": "gpt-test-codex",
        "model_revision": "codex-cli test",
        "base_url": "unused",
        "authentication": "required",
        "response_mode": "strict_json_schema",
        "temperature": 0.0,
        "prompt_hashes": {"codex_batch_v1": "b" * 64},
        "policy_catalogue_hashes": {"banking": "c" * 64},
        "codex_cli_version": "codex-cli test",
        "records_per_batch": 10,
        "reasoning_effort": "high",
        "output_schema_sha256": "d" * 64,
        "taxonomy_version": "1.0",
    }
    original = generation._acquisition_fingerprint(**common)
    for key, value in (
        ("model", "gpt-other"),
        ("codex_cli_version", "codex-cli other"),
        ("records_per_batch", 5),
        ("reasoning_effort", "medium"),
        ("output_schema_sha256", "e" * 64),
        ("taxonomy_version", "other"),
        ("selection_manifest_sha256", "f" * 64),
    ):
        changed = {**common, key: value}
        assert original != generation._acquisition_fingerprint(**changed)


def test_resume_rejects_context_tampering_even_with_a_recomputed_self_hash(
    tiny_environment: _TinyEnvironment,
) -> None:
    output = tiny_environment.root / "release"
    generate_policybench(CONFIG_PATH, output=output, options=_tiny_options())
    artifact = next((tiny_environment.root / "data" / "generated" / "accepted").rglob("*.json"))
    value = json.loads(artifact.read_text("utf-8"))
    value["blueprint_sha256"] = "0" * 64
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = sha256_json(unsigned)
    artifact.write_bytes(canonical_json_bytes(value) + b"\n")

    with pytest.raises(
        PolicyBenchGenerationError,
        match="context mismatch|blueprint_sha256",
    ):
        generate_policybench(
            CONFIG_PATH,
            output=output,
            options=_tiny_options(resume=True),
        )


def test_resume_rejects_artifacts_from_an_incompatible_model(
    tiny_environment: _TinyEnvironment,
) -> None:
    output = tiny_environment.root / "release"
    generate_policybench(
        CONFIG_PATH,
        output=output,
        options=_tiny_options(model="mock-model-a"),
    )
    with pytest.raises(PolicyBenchGenerationError, match="fingerprint|context mismatch"):
        generate_policybench(
            CONFIG_PATH,
            output=output,
            options=_tiny_options(resume=True, model="mock-model-b"),
        )


def test_resume_rebuilds_interrupted_packaging_from_artifacts_byte_identically(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tiny_environment.root / "release"
    generated = generate_policybench(CONFIG_PATH, output=output, options=_tiny_options())
    original_files = _files(output)

    # records.jsonl is written near the start of release assembly.  A missing
    # later report plus a valid intermediate checksum subset models interruption
    # before the final checksum index is atomically installed.
    (output / "generation_report.json").unlink()
    generation.write_named_checksums(
        output,
        ["quality_report.json", "quality_report.md"],
    )

    def unexpected_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("packaging-only resume must not create a provider")

    monkeypatch.setattr(generation, "_create_provider", unexpected_provider)
    resumed = generate_policybench(
        CONFIG_PATH,
        output=output,
        options=_tiny_options(resume=True),
    )

    assert generated.plan_sha256 == resumed.plan_sha256
    assert resumed.reused_accepted_artifacts == 2
    assert _files(output) == original_files
    assert (
        validate_release_directory(output, iter_dataset_records(output))["validation_status"]
        == "PASS"
    )


def test_resume_complete_release_fast_path_never_creates_a_provider(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tiny_environment.root / "release"
    generate_policybench(CONFIG_PATH, output=output, options=_tiny_options())
    original_files = _files(output)

    def unexpected_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("complete-release resume must not create a provider")

    monkeypatch.setattr(generation, "_create_provider", unexpected_provider)
    resumed = generate_policybench(
        CONFIG_PATH,
        output=output,
        options=_tiny_options(resume=True),
    )

    assert resumed.reused_accepted_artifacts == 2
    assert _files(output) == original_files


def test_offline_rebuild_from_accepted_artifacts_is_byte_identical(
    tiny_environment: _TinyEnvironment,
) -> None:
    first_output = tiny_environment.root / "release_first"
    second_output = tiny_environment.root / "release_offline"
    first = generate_policybench(
        CONFIG_PATH,
        output=first_output,
        options=_tiny_options(),
    )
    rebuilt = generate_policybench(
        CONFIG_PATH,
        output=second_output,
        options=_tiny_options(offline=True),
    )

    assert first.automatic_gold_records == rebuilt.automatic_gold_records == 0
    assert rebuilt.reused_accepted_artifacts == 2
    assert _files(first_output) == _files(second_output)
    assert (
        validate_release_directory(second_output, iter_dataset_records(second_output))[
            "validation_status"
        ]
        == "PASS"
    )
    records = [
        json.loads(line)
        for line in (second_output / "records.jsonl").read_text("utf-8").splitlines()
    ]
    assert all(
        record["extensions"]["policybench_v0_1"]["data_quality"] == "SILVER_VALIDATED"
        for record in records
    )
    assert all(
        record["extensions"]["policybench_v0_1"]["human_validation_status"] == "PENDING"
        for record in records
    )
    assert "GOLD" not in repr(records)


def test_offline_mode_never_invokes_even_a_mock_provider_when_artifacts_are_missing(
    tiny_environment: _TinyEnvironment,
) -> None:
    provider = _SequenceProvider(["valid", "valid"])

    with pytest.raises(PolicyBenchGenerationError, match="missing accepted artifact"):
        generate_policybench(
            CONFIG_PATH,
            output=tiny_environment.root / "release_offline_missing",
            options=_tiny_options(offline=True),
            generation_provider=provider,
        )

    assert provider.calls == 0


def test_interrupted_resume_refuses_to_regenerate_a_validated_record_with_missing_artifact(
    tiny_environment: _TinyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tiny_environment.root / "release"
    generate_policybench(CONFIG_PATH, output=output, options=_tiny_options())
    (output / "checksums.sha256").unlink()
    artifact = next((tiny_environment.root / "data" / "generated" / "accepted").rglob("*.json"))
    artifact.unlink()

    def unexpected_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("validated records must never fall back to a provider")

    monkeypatch.setattr(generation, "_create_provider", unexpected_provider)

    with pytest.raises(
        PolicyBenchGenerationError,
        match="missing its accepted generation artifact",
    ):
        generate_policybench(
            CONFIG_PATH,
            output=output,
            options=_tiny_options(resume=True),
        )


def _counterfactual_policy(
    domain: str,
    index: int,
    *,
    allow: bool,
) -> PolicyDescriptor:
    governed_actions = ("READ_FILE", "SEND_MONEY")
    authority_roles = (
        "DEVELOPER",
        "EXTERNAL_CONTENT",
        "MEMORY",
        "SYSTEM",
        "TOOL",
        "UNKNOWN",
    )
    prefix = f"{domain.upper()}_{index:03d}"
    return PolicyDescriptor(
        policy_id=f"{domain}_policy_{index:03d}",
        domain=domain,
        policy_family=f"{domain}_counterfactual_family",
        protected_policy_text_en=f"Synthetic {domain} policy {index} for invariant tests.",
        protected_policy_text_fr=f"Politique synthetique {domain} {index} pour les tests.",
        allowed_actions=governed_actions if allow else ("SEARCH_WEB",),
        forbidden_actions=() if allow else governed_actions,
        confirmation_actions=(),
        rule_ids=(f"{prefix}_ACTION", f"{prefix}_AUTHORITY"),
        rules=(
            {
                "rule_id": f"{prefix}_ACTION",
                "effect": "ALLOW" if allow else "DENY",
                "actions": list(governed_actions),
            },
            {
                "rule_id": f"{prefix}_AUTHORITY",
                "effect": "ALLOW_AUTHORITY" if allow else "DENY_AUTHORITY",
                "actions": list(governed_actions),
                "source_roles": list(authority_roles),
            },
        ),
    )


@pytest.fixture(scope="module")
def counterfactual_plan_inputs() -> tuple[
    PolicyBenchConfig, tuple[PolicyDescriptor, ...], BlueprintPlan
]:
    raw = yaml.safe_load(CONFIG_PATH.read_text("utf-8"))
    raw["target_records"] = 400
    raw["domains"] = {domain: 67 if index < 4 else 66 for index, domain in enumerate(DOMAIN_ORDER)}
    raw["review_sample_size"] = 40
    config = PolicyBenchConfig.from_mapping(raw, environ={})
    policies = tuple(
        _counterfactual_policy(domain, index, allow=allow)
        for domain in DOMAIN_ORDER
        for index, allow in ((1, False), (2, True))
    )
    return config, policies, build_blueprint_plan(config, policies)


def test_counterfactual_donor_plan_is_deterministic_and_preserves_record_quota(
    counterfactual_plan_inputs: tuple[
        PolicyBenchConfig, tuple[PolicyDescriptor, ...], BlueprintPlan
    ],
) -> None:
    config, policies, base_plan = counterfactual_plan_inputs
    first_plan, first_groups = generation._counterfactual_plan(base_plan, config, policies)
    second_plan, second_groups = generation._counterfactual_plan(base_plan, config, policies)

    assert first_plan.sha256() == second_plan.sha256()
    assert [group.group_id for group in first_groups] == [group.group_id for group in second_groups]
    assert len(first_plan.blueprints) == config.target_records
    assert len({item.scenario_id for item in first_plan.blueprints}) == config.target_records
    assert sum(item.counterfactual is not None for item in first_plan.blueprints) == sum(
        config.counterfactual_counts.values()
    )
    record_counts = Counter(
        item.counterfactual.counterfactual_type
        for item in first_plan.blueprints
        if item.counterfactual is not None
    )
    assert record_counts == Counter(config.counterfactual_counts)
    assert len(first_groups) == sum(config.counterfactual_counts.values()) // 2
    assert len({group.parent_scenario_id for group in first_groups}) == len(first_groups)
    assert all(group.validate() == [] for group in first_groups)

    provider = MockGenerationProvider()
    for counterfactual_type in config.counterfactual_counts:
        group = next(
            item for item in first_groups if item.counterfactual_type == counterfactual_type
        )
        parent, sibling = group.members
        responses = []
        for blueprint in (parent, sibling):
            response = provider.generate(
                GenerationRequest(
                    request_id=blueprint.scenario_id,
                    system_prompt="Generate inert research data only.",
                    user_prompt="Realize this closed blueprint.",
                    blueprint=blueprint.to_dict(),
                    protected_policy=blueprint.protected_policy_text,
                    generated_at="2026-07-15T00:00:00Z",
                    seed=blueprint.generation_seed,
                    temperature=0.0,
                )
            )
            responses.append(response)
        sibling_realization = generation._canonicalize_sibling(
            responses[1].data,
            responses[0].data,
            sibling,
        )
        accepted = (
            generation._AcceptedGeneration(parent, dict(responses[0].data), None),
            generation._AcceptedGeneration(sibling, sibling_realization, None),
        )
        generation._validate_realized_unit(accepted)
        if counterfactual_type == "PRESENTATION_CHANGE":
            assert (
                accepted[0].realization["candidate_text"]
                != accepted[1].realization["candidate_text"]
            )
        else:
            assert (
                accepted[0].realization["candidate_text"]
                == accepted[1].realization["candidate_text"]
            )
