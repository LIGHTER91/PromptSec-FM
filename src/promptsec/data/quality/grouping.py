"""Explainable template-family grouping from source provenance only.

Grouping deliberately never inspects ``content.text``.  Every automatic rule
below is tied to metadata preserved by an importer; unknown provenance is sent
to manual review instead of being guessed from keywords.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

_PROMPTINJECT_FAMILIES = {
    # Preserve the typo in the published upstream collection identifier.
    "goal_hikacking_attacks": "override_previous_instructions",
    "goal_hijacking_attacks": "override_previous_instructions",
    "prompt_leaking_attacks": "prompt_or_policy_disclosure",
}

_NOTINJECT_FAMILIES = {
    "Technique Queries": "quoted_attack_hard_negative",
    "Common Queries": "notinject_common_query",
    "Virtual Creation": "notinject_virtual_creation",
    "Multilingual": "notinject_multilingual_hard_negative",
}

# Documented mappings from BIPIA's pinned attack-domain metadata.  These are
# provenance rules, not text classifiers.  Remaining known domains receive a
# stable ``bipia_<slug>`` family below.
_BIPIA_TOOL_EXFILTRATION = {
    ("Cookie Theft", "code"),
    ("Data Eavesdropping", "code"),
    ("Dumpster Diving", "code"),
    ("Environment Variable Analysis", "code"),
    ("Keylogging", "code"),
    ("Memory Scanning", "code"),
    ("Screen Scraping", "code"),
    ("Traffic Analysis", "code"),
}
_BIPIA_SYSTEM_COMPROMISE = {
    ("Blocking Internet Connection", "code"),
    ("Bringing Down Hosts and Servers (Denial of Service)", "code"),
    ("Compromising Computers", "code"),
    ("Corrupting an Operating System", "code"),
    ("Crippling Critical Infrastructures", "code"),
    ("Cryptocurrency Mining", "code"),
    ("Encrypting Documents and Demanding Ransom (Ransomware)", "code"),
    ("Exploiting System Vulnerabilities", "code"),
    ("Malware Distribution", "text"),
    ("Network Propagation", "code"),
}


def assign_group(record: Mapping[str, Any]) -> dict[str, Any]:
    """Assign a documented template family using only record provenance.

    The returned object always contains ``domain``, ``template_family``,
    ``assignment_method``, ``assignment_rule`` and ``requires_manual_review``.
    """

    source_id, original_fields, original_labels = _source_metadata(record)
    normalized_source = _slug(source_id)

    if normalized_source == "promptinject":
        collection = _string_value("collection", original_fields, original_labels)
        key = _string_value("key", original_fields, original_labels)
        family = _PROMPTINJECT_FAMILIES.get(collection or "")
        if collection and key and family:
            return _assignment(
                domain=collection,
                template_family=family,
                assignment_method="DOCUMENTED_RULE",
                assignment_rule=f"promptinject.collection_key:{collection}:{key}",
            )
        return _manual_assignment(
            domain=collection or source_id,
            rule="promptinject.missing_or_unknown_collection_key",
        )

    if normalized_source == "open_prompt_injection":
        task = _string_value("task", original_fields, original_labels)
        template_name = _string_value("template_name", original_fields, original_labels)
        if task and template_name:
            return _assignment(
                domain=task,
                template_family=f"open_pi_{_slug(task)}",
                assignment_method="GENERATION_TEMPLATE",
                assignment_rule=f"open_pi.task_template:{task}:{template_name}",
            )
        return _manual_assignment(
            domain=task or source_id,
            rule="open_pi.missing_task_or_template_name",
        )

    if normalized_source == "notinject":
        category = _string_value("category", original_fields, original_labels)
        family = _NOTINJECT_FAMILIES.get(category or "")
        if category and family:
            return _assignment(
                domain=category,
                template_family=family,
                assignment_method="SOURCE_METADATA",
                assignment_rule=f"notinject.category:{category}",
            )
        return _manual_assignment(
            domain=category or source_id,
            rule="notinject.missing_or_unknown_category",
        )

    if normalized_source == "bipia":
        attack_domain = _string_value("attack_domain", original_fields, original_labels)
        payload_type = _string_value("payload_type", original_fields, original_labels)
        if not attack_domain or not payload_type:
            return _manual_assignment(
                domain=attack_domain or source_id,
                rule="bipia.missing_attack_domain_or_payload_type",
            )

        source_pair = (attack_domain, payload_type)
        if source_pair in _BIPIA_TOOL_EXFILTRATION:
            family = "tool_exfiltration"
            rule_name = "tool_exfiltration_table"
        elif source_pair in _BIPIA_SYSTEM_COMPROMISE:
            family = "system_compromise"
            rule_name = "system_compromise_table"
        else:
            family = f"bipia_{_slug(attack_domain)}"
            rule_name = "documented_domain_slug_fallback"
        return _assignment(
            domain=attack_domain,
            template_family=family,
            assignment_method="DOCUMENTED_RULE",
            assignment_rule=(f"bipia.{rule_name}:{attack_domain}:{payload_type}"),
        )

    return _manual_assignment(
        domain=source_id,
        rule="fallback.no_documented_source_provenance_rule",
    )


def _source_metadata(
    record: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    metadata = record.get("metadata")
    provenance = metadata.get("dataset_provenance") if isinstance(metadata, Mapping) else None
    source_dataset = provenance.get("source_dataset") if isinstance(provenance, Mapping) else None
    source_record = provenance.get("source_record") if isinstance(provenance, Mapping) else None
    source_id_value = source_dataset.get("id") if isinstance(source_dataset, Mapping) else None
    source_id = (
        source_id_value if isinstance(source_id_value, str) and source_id_value else "unknown"
    )
    original_fields = (
        source_record.get("original_fields") if isinstance(source_record, Mapping) else None
    )
    original_labels = (
        source_record.get("original_labels") if isinstance(source_record, Mapping) else None
    )
    return (
        source_id,
        original_fields if isinstance(original_fields, Mapping) else {},
        original_labels if isinstance(original_labels, Mapping) else {},
    )


def _string_value(
    key: str,
    original_fields: Mapping[str, Any],
    original_labels: Mapping[str, Any],
) -> str | None:
    for source in (original_fields, original_labels):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _assignment(
    *,
    domain: str,
    template_family: str,
    assignment_method: str,
    assignment_rule: str,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "template_family": template_family,
        "assignment_method": assignment_method,
        "assignment_rule": assignment_rule,
        "requires_manual_review": False,
    }


def _manual_assignment(*, domain: str, rule: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "template_family": "manual_review",
        "assignment_method": "MANUAL_REVIEW",
        "assignment_rule": rule,
        "requires_manual_review": True,
    }


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_") or "unknown"


__all__ = ["assign_group"]
