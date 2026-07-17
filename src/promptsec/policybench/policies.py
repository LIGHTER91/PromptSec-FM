"""Loading and semantic validation for PolicyBench policy catalogues."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from promptsec.policybench.schemas import validate_policy_catalog_schema

SUPPORTED_DOMAINS = frozenset(
    {
        "banking",
        "email",
        "calendar",
        "file_management",
        "web_and_purchases",
        "persistent_memory",
    }
)

SUPPORTED_ACTIONS = frozenset(
    {
        "ANALYZE_CONTENT",
        "ADD_PAYEE",
        "ADD_TO_CART",
        "BROWSE_WEB",
        "CALL_API",
        "CANCEL_TRANSACTION",
        "CHANGE_FILE_PERMISSIONS",
        "CHANGE_SECURITY_SETTINGS",
        "COPY_FILE",
        "CREATE_ACCOUNT",
        "CREATE_EVENT",
        "CREATE_FILE",
        "DELETE_EMAIL",
        "DELETE_EVENT",
        "DELETE_FILE",
        "DELETE_MEMORY",
        "DOWNLOAD_ATTACHMENT",
        "DOWNLOAD_FILE",
        "EXECUTE_CODE",
        "EXPORT_FINANCIAL_DATA",
        "EXPORT_MEMORY",
        "FORWARD_EMAIL",
        "IMPORT_MEMORY",
        "INVITE_ATTENDEE",
        "LIST_FILES",
        "MODIFY_FILE",
        "MOVE_FILE",
        "OPEN_ATTACHMENT",
        "PURCHASE_ITEM",
        "READ_ACCOUNT_BALANCE",
        "READ_CALENDAR",
        "READ_EMAIL",
        "READ_FILE",
        "READ_MEMORY",
        "READ_TRANSACTION_HISTORY",
        "READ_WEBPAGE",
        "REMOVE_PAYEE",
        "RENAME_FILE",
        "REPLY_EMAIL",
        "REVEAL_SECRET",
        "REVEAL_SYSTEM_PROMPT",
        "RUN_COMMAND",
        "SAVE_PAYMENT_METHOD",
        "SCHEDULE_TRANSACTION",
        "SEARCH_CALENDAR",
        "SEARCH_EMAIL",
        "SEARCH_FILES",
        "SEARCH_MEMORY",
        "SEARCH_WEB",
        "SEND_EMAIL",
        "SEND_MONEY",
        "SHARE_CALENDAR",
        "SHARE_FILE",
        "SHARE_MEMORY",
        "STORE_MEMORY",
        "SUBMIT_PAYMENT",
        "SUMMARIZE_EMAIL",
        "UPDATE_EVENT",
        "UPDATE_MEMORY",
        "UPLOAD_FILE",
    }
)

_STRUCTURED_RULE_EFFECTS = frozenset({"ALLOW", "ALLOW_IF", "DENY", "DENY_IF", "DENY_UNLESS"})
_AUTHORITY_RULE_EFFECTS = frozenset({"ALLOW_AUTHORITY", "DENY_AUTHORITY", "DELEGATE_AUTHORITY"})
_SOURCE_ROLES = frozenset(
    {"SYSTEM", "DEVELOPER", "USER", "ASSISTANT", "TOOL", "EXTERNAL_CONTENT", "MEMORY", "UNKNOWN"}
)


class PolicyCatalogError(ValueError):
    """Raised when a policy catalogue is structurally or semantically invalid."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        if not self.errors:
            raise ValueError("PolicyCatalogError requires at least one error")
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"Policy catalogue validation failed:\n{details}")


def _duplicates(values: Sequence[Any]) -> list[str]:
    counts = Counter(value for value in values if isinstance(value, str))
    return sorted(value for value, count in counts.items() if count > 1)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _actions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _unknown_action_errors(actions: Sequence[str], path: str) -> list[str]:
    return [
        f"{path}: unknown action {action!r}"
        for action in sorted(set(actions).difference(SUPPORTED_ACTIONS))
    ]


def _rule_ids(policy: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("structured_rules", "confirmation_requirements", "source_authority_rules"):
        rules = policy.get(field)
        if not isinstance(rules, list):
            continue
        values.extend(
            rule_id
            for rule in rules
            if isinstance(rule, Mapping) and isinstance((rule_id := rule.get("rule_id")), str)
        )
    return values


def _policy_errors(policy: Mapping[str, Any], index: int, catalog_domain: Any) -> list[str]:
    policy_id = policy.get("policy_id")
    label = policy_id if _nonempty_string(policy_id) else f"index {index}"
    path = f"$.policies[{index}] ({label})"
    errors: list[str] = []

    if policy.get("domain") != catalog_domain:
        errors.append(f"{path}.domain: must match catalogue domain {catalog_domain!r}")

    title = policy.get("title")
    if not isinstance(title, Mapping):
        errors.append(f"{path}.title: expected bilingual title object")
    else:
        for language in ("en", "fr"):
            if not _nonempty_string(title.get(language)):
                errors.append(f"{path}.title.{language}: must be non-empty")

    for field in ("protected_policy_text_en", "protected_policy_text_fr"):
        if not _nonempty_string(policy.get(field)):
            errors.append(f"{path}.{field}: must be non-empty")

    allowed_actions = _actions(policy.get("allowed_actions"))
    forbidden_actions = _actions(policy.get("forbidden_actions"))
    errors.extend(_unknown_action_errors(allowed_actions, f"{path}.allowed_actions"))
    errors.extend(_unknown_action_errors(forbidden_actions, f"{path}.forbidden_actions"))
    overlap = sorted(set(allowed_actions).intersection(forbidden_actions))
    if overlap:
        errors.append(f"{path}: actions cannot be both globally allowed and forbidden: {overlap}")

    structured_rules = policy.get("structured_rules")
    structured_rules = structured_rules if isinstance(structured_rules, list) else []
    confirmation_requirements = policy.get("confirmation_requirements")
    confirmation_requirements = (
        confirmation_requirements if isinstance(confirmation_requirements, list) else []
    )
    authority_rules = policy.get("source_authority_rules")
    authority_rules = authority_rules if isinstance(authority_rules, list) else []
    if not authority_rules:
        errors.append(f"{path}.source_authority_rules: at least one authority boundary is required")

    all_rule_ids = [
        rule.get("rule_id")
        for rule in [*structured_rules, *confirmation_requirements, *authority_rules]
        if isinstance(rule, Mapping)
    ]
    duplicates = _duplicates(all_rule_ids)
    if duplicates:
        errors.append(f"{path}: duplicate rule IDs: {duplicates}")

    unconditional: dict[str, set[str]] = {}
    for rule_index, rule in enumerate(structured_rules):
        if not isinstance(rule, Mapping):
            continue
        rule_path = f"{path}.structured_rules[{rule_index}]"
        effect = rule.get("effect")
        if effect not in _STRUCTURED_RULE_EFFECTS:
            errors.append(f"{rule_path}.effect: unsupported value {effect!r}")
        action = rule.get("action")
        if isinstance(action, str):
            errors.extend(_unknown_action_errors([action], f"{rule_path}.action"))
        conditions = rule.get("conditions")
        if effect in {"ALLOW", "DENY"} and (conditions is None or conditions == {}):
            unconditional.setdefault(str(action), set()).add(str(effect))

    contradictions = sorted(
        action for action, effects in unconditional.items() if {"ALLOW", "DENY"} <= effects
    )
    if contradictions:
        errors.append(
            f"{path}.structured_rules: contradictory unconditional ALLOW/DENY "
            f"rules for {contradictions}"
        )
    denied_but_listed_allowed = sorted(
        action for action in allowed_actions if "DENY" in unconditional.get(action, set())
    )
    allowed_but_listed_forbidden = sorted(
        action for action in forbidden_actions if "ALLOW" in unconditional.get(action, set())
    )
    if denied_but_listed_allowed:
        errors.append(
            f"{path}: allowed_actions contradict unconditional DENY rules for "
            f"{denied_but_listed_allowed}"
        )
    if allowed_but_listed_forbidden:
        errors.append(
            f"{path}: forbidden_actions contradict unconditional ALLOW rules for "
            f"{allowed_but_listed_forbidden}"
        )

    authority_effects: dict[tuple[str, str], set[str]] = {}
    for rule_index, rule in enumerate(authority_rules):
        if not isinstance(rule, Mapping):
            continue
        rule_path = f"{path}.source_authority_rules[{rule_index}]"
        effect = rule.get("effect")
        if effect not in _AUTHORITY_RULE_EFFECTS:
            errors.append(f"{rule_path}.effect: unsupported value {effect!r}")
        errors.extend(_unknown_action_errors(_actions(rule.get("actions")), f"{rule_path}.actions"))
        roles = _actions(rule.get("source_roles"))
        for role in sorted(set(roles).difference(_SOURCE_ROLES)):
            errors.append(f"{rule_path}.source_roles: unknown source role {role!r}")
        for role in roles:
            for action in _actions(rule.get("actions")):
                authority_effects.setdefault((role, action), set()).add(str(effect))

    authority_contradictions = sorted(
        f"{role}/{action}"
        for (role, action), effects in authority_effects.items()
        if {"ALLOW_AUTHORITY", "DENY_AUTHORITY"} <= effects
    )
    if authority_contradictions:
        errors.append(
            f"{path}.source_authority_rules: contradictory unconditional authority "
            f"rules for {authority_contradictions}"
        )

    for requirement_index, requirement in enumerate(confirmation_requirements):
        if not isinstance(requirement, Mapping):
            continue
        requirement_path = f"{path}.confirmation_requirements[{requirement_index}]"
        action = requirement.get("action")
        if isinstance(action, str):
            errors.extend(_unknown_action_errors([action], f"{requirement_path}.action"))

    return errors


def validate_policy_catalog(catalog: Any) -> list[str]:
    """Return schema and semantic errors for one in-memory catalogue."""

    errors = validate_policy_catalog_schema(catalog)
    if not isinstance(catalog, Mapping):
        return errors

    domain = catalog.get("domain")
    if domain not in SUPPORTED_DOMAINS:
        errors.append(f"$.domain: unsupported domain {domain!r}")
    policies = catalog.get("policies")
    if not isinstance(policies, list):
        return list(dict.fromkeys(errors))

    policy_ids = [policy.get("policy_id") for policy in policies if isinstance(policy, Mapping)]
    duplicates = _duplicates(policy_ids)
    if duplicates:
        errors.append(f"$.policies: duplicate policy IDs: {duplicates}")

    all_rule_ids = [
        rule_id
        for policy in policies
        if isinstance(policy, Mapping)
        for rule_id in _rule_ids(policy)
    ]
    duplicate_rule_ids = _duplicates(all_rule_ids)
    if duplicate_rule_ids:
        errors.append(f"$.policies: duplicate rule IDs across policies: {duplicate_rule_ids}")

    for index, policy in enumerate(policies):
        if isinstance(policy, Mapping):
            errors.extend(_policy_errors(policy, index, domain))

    return list(dict.fromkeys(errors))


def require_valid_policy_catalog(catalog: Any) -> None:
    errors = validate_policy_catalog(catalog)
    if errors:
        raise PolicyCatalogError(errors)


def load_policy_catalog(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML catalogue and require it to be valid."""

    catalog_path = Path(path)
    try:
        value = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"policy catalogue not found: {catalog_path}") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PolicyCatalogError([f"{catalog_path}: cannot decode YAML: {error}"]) from error
    require_valid_policy_catalog(value)
    if not isinstance(value, dict):  # Narrowing after the validator for type checkers.
        raise PolicyCatalogError([f"{catalog_path}: expected an object"])
    return value


def load_policy_catalogs(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Load a complete six-domain catalogue directory with global ID checks."""

    root = Path(directory)
    paths = sorted(
        {path for pattern in ("*.yaml", "*.yml") for path in root.glob(pattern)},
        key=lambda path: path.name,
    )
    if not paths:
        raise PolicyCatalogError([f"{root}: no YAML policy catalogues found"])

    catalogues: dict[str, dict[str, Any]] = {}
    policy_locations: dict[str, list[str]] = {}
    rule_locations: dict[str, list[str]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            catalogue = load_policy_catalog(path)
        except PolicyCatalogError as error:
            errors.extend(f"{path.name}: {item}" for item in error.errors)
            continue
        domain = str(catalogue["domain"])
        if domain in catalogues:
            errors.append(f"duplicate catalogue domain {domain!r}: {path.name}")
            continue
        catalogues[domain] = catalogue
        for policy in catalogue["policies"]:
            policy_id = str(policy["policy_id"])
            policy_locations.setdefault(policy_id, []).append(path.name)
            for rule_id in _rule_ids(policy):
                rule_locations.setdefault(rule_id, []).append(path.name)

    for policy_id, locations in sorted(policy_locations.items()):
        if len(locations) > 1:
            errors.append(f"duplicate policy ID {policy_id!r} across {sorted(locations)}")
    for rule_id, locations in sorted(rule_locations.items()):
        if len(locations) > 1:
            errors.append(f"duplicate rule ID {rule_id!r} across {sorted(locations)}")

    missing = sorted(SUPPORTED_DOMAINS.difference(catalogues))
    extra = sorted(set(catalogues).difference(SUPPORTED_DOMAINS))
    if missing:
        errors.append(f"missing policy catalogue domains: {missing}")
    if extra:
        errors.append(f"unsupported policy catalogue domains: {extra}")
    if errors:
        raise PolicyCatalogError(errors)
    return dict(sorted(catalogues.items()))


# British-spelling aliases are convenient in documentation without creating a
# second implementation.
load_policy_catalogue = load_policy_catalog
load_policy_catalogues = load_policy_catalogs
validate_policy_catalogue = validate_policy_catalog


__all__ = [
    "PolicyCatalogError",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_DOMAINS",
    "load_policy_catalog",
    "load_policy_catalogs",
    "load_policy_catalogue",
    "load_policy_catalogues",
    "require_valid_policy_catalog",
    "validate_policy_catalog",
    "validate_policy_catalogue",
]
