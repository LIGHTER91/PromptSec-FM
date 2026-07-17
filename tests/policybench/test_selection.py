from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from promptsec.data.hashing import sha256_file
from promptsec.policybench.blueprints import (
    BlueprintPlan,
    build_blueprint_plan,
    policy_descriptors_from_catalogues,
)
from promptsec.policybench.config import (
    CATEGORY_ORDER,
    COUNTERFACTUAL_TYPE_ORDER,
    DOMAIN_ORDER,
    PolicyBenchConfig,
)
from promptsec.policybench.generation import _counterfactual_plan
from promptsec.policybench.policies import load_policy_catalogs
from promptsec.policybench.selection import (
    PolicyBenchSelectionError,
    build_selection_manifest,
    load_selection_manifest,
    select_stratified_blueprints,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "policybench_codex_pilot_100_v0.1.yaml"
POLICY_ROOT = ROOT / "data" / "policybench" / "policies"


@pytest.fixture(scope="module")
def selection_inputs() -> tuple[BlueprintPlan, tuple, dict]:
    config = PolicyBenchConfig.load(
        CONFIG_PATH,
        environ={"PROMPTSEC_CODEX_MODEL": "gpt-5.6-sol"},
    )
    catalogues = load_policy_catalogs(POLICY_ROOT)
    policies = policy_descriptors_from_catalogues(catalogues)
    base_plan = build_blueprint_plan(config, policies)
    plan, _ = _counterfactual_plan(base_plan, config, policies)
    selected = select_stratified_blueprints(
        plan,
        seed=20260715,
        records_per_category=10,
    )
    manifest = build_selection_manifest(
        plan,
        selected,
        seed=20260715,
        records_per_category=10,
        source_config_sha256=sha256_file(CONFIG_PATH),
    )
    return plan, selected, manifest


def test_stratified_selection_is_exact_deterministic_and_complete(
    selection_inputs: tuple[BlueprintPlan, tuple, dict],
) -> None:
    plan, selected, manifest = selection_inputs
    repeated = select_stratified_blueprints(
        plan,
        seed=20260715,
        records_per_category=10,
    )

    assert selected == repeated
    assert len(selected) == len({item.scenario_id for item in selected}) == 100
    assert Counter(item.category for item in selected) == Counter(
        {category: 10 for category in CATEGORY_ORDER}
    )
    assert set(item.domain for item in selected) == set(DOMAIN_ORDER)
    assert set(item.language for item in selected) == {"en", "fr"}
    assert set(manifest["counterfactual_type_counts"]) == set(COUNTERFACTUAL_TYPE_ORDER)
    assert set(manifest["policy_alignment_expectations"]) == {
        "COMPLIANT",
        "CONFLICTING",
        "NOT_APPLICABLE",
        "UNDETERMINED",
    }
    assert set(manifest["authority_status_expectations"]) == {
        "NOT_APPLICABLE",
        "OUTSIDE_AUTHORITY",
        "SPOOFED",
        "UNKNOWN",
        "WITHIN_AUTHORITY",
    }
    selected_ids = {item.scenario_id for item in selected}
    source_groups: dict[str, set[str]] = {}
    for blueprint in plan.blueprints:
        if blueprint.counterfactual is not None:
            source_groups.setdefault(blueprint.counterfactual.counterfactual_group_id, set()).add(
                blueprint.scenario_id
            )
    assert all(
        selected_ids.isdisjoint(group) or group <= selected_ids for group in source_groups.values()
    )


def test_selection_manifest_round_trip_rejects_metadata_tampering(
    selection_inputs: tuple[BlueprintPlan, tuple, dict],
    tmp_path: Path,
) -> None:
    plan, selected, manifest = selection_inputs
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, digest = load_selection_manifest(
        path,
        plan,
        seed=20260715,
        source_config_sha256=sha256_file(CONFIG_PATH),
    )
    assert loaded == selected
    assert digest == sha256_file(path)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    first_domain = next(iter(tampered["domain_counts"]))
    tampered["domain_counts"][first_domain] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PolicyBenchSelectionError, match="metadata does not match"):
        load_selection_manifest(
            path,
            plan,
            seed=20260715,
            source_config_sha256=sha256_file(CONFIG_PATH),
        )


def test_selection_manifest_rejects_partial_counterfactual_groups(
    selection_inputs: tuple[BlueprintPlan, tuple, dict],
) -> None:
    plan, selected, _ = selection_inputs
    counterfactual = next(item for item in selected if item.counterfactual is not None)
    incomplete = tuple(item for item in selected if item.scenario_id != counterfactual.scenario_id)
    with pytest.raises(PolicyBenchSelectionError, match="splits counterfactual"):
        build_selection_manifest(
            plan,
            incomplete,
            seed=20260715,
            records_per_category=10,
            source_config_sha256=sha256_file(CONFIG_PATH),
        )
