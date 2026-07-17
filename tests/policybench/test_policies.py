from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from promptsec.policybench.policies import (
    SUPPORTED_ACTIONS,
    SUPPORTED_DOMAINS,
    PolicyCatalogError,
    load_policy_catalogs,
    require_valid_policy_catalog,
    validate_policy_catalog,
)
from promptsec.policybench.schemas import POLICY_CATALOG_SCHEMA, load_schema

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIRECTORY = REPOSITORY_ROOT / "data" / "policybench" / "policies"


def _catalogue(domain: str = "banking") -> dict:
    value = yaml.safe_load((POLICY_DIRECTORY / f"{domain}.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_all_six_catalogues_are_valid_bilingual_and_globally_unique() -> None:
    catalogues = load_policy_catalogs(POLICY_DIRECTORY)

    assert set(catalogues) == SUPPORTED_DOMAINS
    assert {domain: len(value["policies"]) for domain, value in catalogues.items()} == {
        domain: 20 for domain in SUPPORTED_DOMAINS
    }
    policy_ids = []
    for domain, catalogue in catalogues.items():
        for policy in catalogue["policies"]:
            assert policy["domain"] == domain
            assert policy["title"]["en"].strip()
            assert policy["title"]["fr"].strip()
            assert policy["protected_policy_text_en"].strip()
            assert policy["protected_policy_text_fr"].strip()
            assert policy["source_authority_rules"]
            policy_ids.append(policy["policy_id"])
    assert len(policy_ids) == len(set(policy_ids)) == 120


def test_python_action_vocabulary_matches_the_strict_schema() -> None:
    schema = load_schema(POLICY_CATALOG_SCHEMA)

    assert set(schema["$defs"]["action"]["enum"]) == SUPPORTED_ACTIONS


def test_duplicate_policy_ids_are_rejected() -> None:
    catalogue = _catalogue()
    catalogue["policies"][1]["policy_id"] = catalogue["policies"][0]["policy_id"]

    errors = validate_policy_catalog(catalogue)

    assert any("duplicate policy IDs" in error for error in errors)


@pytest.mark.parametrize("rule_collection", ["confirmation_requirements", "source_authority_rules"])
def test_duplicate_rule_ids_across_rule_kinds_are_rejected(rule_collection: str) -> None:
    catalogue = _catalogue()
    policy = catalogue["policies"][0]
    policy[rule_collection][0]["rule_id"] = policy["structured_rules"][0]["rule_id"]

    errors = validate_policy_catalog(catalogue)

    assert any("duplicate rule IDs" in error for error in errors)


def test_contradictory_unconditional_allow_and_deny_are_rejected() -> None:
    catalogue = _catalogue()
    policy = catalogue["policies"][0]
    denied_action = policy["structured_rules"][1]["action"]
    policy["structured_rules"].append(
        {
            "rule_id": "BANK_CONTRADICTORY_ALLOW_001",
            "effect": "ALLOW",
            "action": denied_action,
            "conditions": {},
        }
    )

    errors = validate_policy_catalog(catalogue)

    assert any("contradictory unconditional ALLOW/DENY" in error for error in errors)


def test_action_lists_cannot_contradict_unconditional_structured_rules() -> None:
    catalogue = _catalogue()
    policy = catalogue["policies"][0]
    action = policy["allowed_actions"][0]
    policy["structured_rules"].append(
        {
            "rule_id": "BANK_LIST_CONTRADICTION_001",
            "effect": "DENY",
            "action": action,
            "conditions": {},
        }
    )

    errors = validate_policy_catalog(catalogue)

    assert any("allowed_actions contradict unconditional DENY" in error for error in errors)


def test_conflicting_source_authority_boundaries_are_rejected() -> None:
    catalogue = _catalogue()
    policy = catalogue["policies"][0]
    existing = policy["source_authority_rules"][0]
    opposite = "DENY_AUTHORITY" if existing["effect"] == "ALLOW_AUTHORITY" else "ALLOW_AUTHORITY"
    policy["source_authority_rules"].append(
        {
            "rule_id": "BANK_AUTHORITY_CONTRADICTION_001",
            "effect": opposite,
            "source_roles": list(existing["source_roles"]),
            "actions": list(existing["actions"]),
        }
    )

    errors = validate_policy_catalog(catalogue)

    assert any("contradictory unconditional authority" in error for error in errors)


def test_unknown_actions_are_rejected_with_a_readable_error() -> None:
    catalogue = _catalogue()
    catalogue["policies"][0]["allowed_actions"][0] = "TELEPORT_FUNDS"

    errors = validate_policy_catalog(catalogue)

    assert any("unknown action 'TELEPORT_FUNDS'" in error for error in errors)


@pytest.mark.parametrize(
    "field",
    ["protected_policy_text_en", "protected_policy_text_fr"],
)
def test_blank_bilingual_policy_text_is_rejected(field: str) -> None:
    catalogue = _catalogue()
    catalogue["policies"][0][field] = "   "

    errors = validate_policy_catalog(catalogue)

    assert any(f"{field}: must be non-empty" in error for error in errors)


def test_missing_source_authority_boundary_is_rejected() -> None:
    catalogue = _catalogue()
    catalogue["policies"][0]["source_authority_rules"] = []

    errors = validate_policy_catalog(catalogue)

    assert any("at least one authority boundary is required" in error for error in errors)


def test_unsupported_schema_values_and_extra_fields_are_rejected() -> None:
    catalogue = _catalogue()
    catalogue["policies"][0]["structured_rules"][0]["effect"] = "PERMIT"
    catalogue["policies"][0]["unexpected"] = True

    errors = validate_policy_catalog(catalogue)

    assert any("PERMIT" in error for error in errors)
    assert any("unexpected" in error for error in errors)


def test_require_valid_policy_catalog_raises_all_semantic_errors() -> None:
    catalogue = copy.deepcopy(_catalogue())
    catalogue["policies"][0]["source_authority_rules"] = []
    catalogue["policies"][0]["protected_policy_text_en"] = ""

    with pytest.raises(PolicyCatalogError) as raised:
        require_valid_policy_catalog(catalogue)

    assert len(raised.value.errors) >= 2
