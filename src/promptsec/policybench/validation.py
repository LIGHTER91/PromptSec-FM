"""Semantic and canonical validation for PromptSec-PolicyBench records."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file, sha256_json, sha256_text
from promptsec.data.validation import validate_record
from promptsec.policybench.io import iter_jsonl, read_json_object, safe_child
from promptsec.policybench.schemas import (
    schema_path,
    validate_generation_response_schema,
    validate_policybench_record_schema,
    validate_scenario_blueprint_schema,
)
from promptsec.policybench.spans import SpanError, resolve_generation_anchors, validate_spans

_URL = re.compile(r"https?://[^\s<>'\"]+", flags=re.IGNORECASE)
_AUTHORITY_MARKER = re.compile(
    r"(?:\[[^\]\r\n]{0,80}\b(?:system|developer|administrator|admin|security\s+team|"
    r"système|développeur|administrateur|équipe\s+de\s+sécurité)\b[^\]\r\n]{0,80}\]|"
    r"(?:system|developer|administrator|security\s+team|équipe\s+de\s+sécurité)\s+"
    r"(?:message|override|instruction|command|commande|prioritaire)|"
    r"(?:message|instruction|commande)\s+(?:système|développeur|administrateur))",
    flags=re.IGNORECASE,
)
_LABEL_LEAK = re.compile(
    r"\b(?:NO_INSTRUCTION|INSTRUCTION_PRESENT|QUOTED_OR_REPORTED|"
    r"OUTSIDE_AUTHORITY|WITHIN_AUTHORITY|PROMPT_INJECTION|TASK_HIJACKING|"
    r"POLICY_BYPASS|SILVER_GENERATED|GOLD_HUMAN_CONFIRMED|"
    r"instruction_presence|protected_policy_alignment|authority_status)\b"
)
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_PUBLISHED_SPLITS = (
    "train",
    "validation",
    "test_policy_family_ood",
    "test_domain_ood",
    "test_language_ood",
    "test_counterfactual",
    "human_review_candidates",
)

_ACTION_VERBS = {
    "ADD": {"en": ("add",), "fr": ("ajout",)},
    "BROWSE": {"en": ("browse", "visit"), "fr": ("navig", "consult", "parcour")},
    "CALL": {"en": ("call",), "fr": ("appel", "invoqu")},
    "CANCEL": {"en": ("cancel",), "fr": ("annul",)},
    "CHANGE": {"en": ("change",), "fr": ("chang", "modifi")},
    "COPY": {"en": ("copy",), "fr": ("copi",)},
    "CREATE": {"en": ("create",), "fr": ("cré",)},
    "DELETE": {"en": ("delete", "remove"), "fr": ("supprim", "effac")},
    "DOWNLOAD": {"en": ("download",), "fr": ("télécharg",)},
    "EXECUTE": {"en": ("execute", "run"), "fr": ("exécut", "lanc")},
    "EXPORT": {"en": ("export",), "fr": ("export",)},
    "FORWARD": {"en": ("forward",), "fr": ("transfér",)},
    "IMPORT": {"en": ("import",), "fr": ("import",)},
    "INVITE": {"en": ("invite",), "fr": ("invit",)},
    "LIST": {"en": ("list",), "fr": ("list", "énumér", "répert")},
    "MODIFY": {"en": ("modify", "edit"), "fr": ("modifi",)},
    "MOVE": {"en": ("move",), "fr": ("déplac",)},
    "OPEN": {"en": ("open",), "fr": ("ouvr",)},
    "PURCHASE": {"en": ("purchase", "buy"), "fr": ("achet", "achèt")},
    "READ": {"en": ("read", "show", "review"), "fr": ("lis", "lire", "consult")},
    "REMOVE": {"en": ("remove",), "fr": ("retir", "supprim")},
    "RENAME": {"en": ("rename",), "fr": ("renomm",)},
    "REPLY": {"en": ("reply", "respond"), "fr": ("répond",)},
    "REVEAL": {
        "en": ("reveal", "show", "disclose"),
        "fr": ("révèl", "révél", "divulgu", "affich"),
    },
    "RUN": {"en": ("run",), "fr": ("exécut", "lanc")},
    "SAVE": {"en": ("save",), "fr": ("enregistr",)},
    "SCHEDULE": {"en": ("schedule",), "fr": ("programm",)},
    "SEARCH": {"en": ("search", "find"), "fr": ("recherch", "trouv")},
    "SEND": {"en": ("send", "transfer"), "fr": ("envoi", "envoy", "transfér")},
    "SHARE": {"en": ("share",), "fr": ("partag",)},
    "STORE": {
        "en": ("store", "remember"),
        "fr": ("mémoris", "stock", "enregistr"),
    },
    "SUBMIT": {"en": ("submit", "pay"), "fr": ("soumet", "pai")},
    "SUMMARIZE": {"en": ("summar",), "fr": ("résum",)},
    "UPDATE": {"en": ("update",), "fr": ("met", "actualis", "modifi")},
    "UPLOAD": {"en": ("upload",), "fr": ("télévers", "import")},
}

_OBJECT_TERMS = {
    "MONEY": {
        "en": ("money", "fund", "payment", "transfer"),
        "fr": ("argent", "fonds", "paiement", "virement"),
    },
    "EMAIL": {"en": ("email", "message"), "fr": ("courriel", "e-mail", "message")},
    "EVENT": {"en": ("event", "meeting"), "fr": ("événement", "réunion")},
    "FILE": {"en": ("file", "document"), "fr": ("fichier", "document")},
    "MEMORY": {"en": ("memory", "remember"), "fr": ("mémoire", "souvenir")},
    "SECRET": {"en": ("secret", "credential"), "fr": ("secret", "identifiant")},
    "SYSTEM_PROMPT": {
        "en": ("system prompt", "hidden instruction"),
        "fr": ("prompt système", "instruction cachée"),
    },
    "TRANSACTION": {"en": ("transaction", "payment"), "fr": ("transaction", "paiement")},
    "CALENDAR": {"en": ("calendar",), "fr": ("calendrier", "agenda")},
    "WEBPAGE": {"en": ("page", "website"), "fr": ("page", "site")},
}

_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_LANGUAGE_MARKERS = {
    "en": frozenset(
        {
            "account",
            "and",
            "balance",
            "contains",
            "dated",
            "example",
            "file",
            "human",
            "ignore",
            "money",
            "must",
            "operator",
            "other",
            "please",
            "previous",
            "quoted",
            "read",
            "requests",
            "research",
            "reveal",
            "send",
            "should",
            "statement",
            "teaching",
            "the",
            "this",
            "three",
            "with",
            "without",
            "would",
        }
    ),
    "fr": frozenset(
        {
            "argent",
            "autres",
            "avec",
            "citée",
            "compte",
            "contient",
            "demandes",
            "des",
            "doit",
            "du",
            "envoyer",
            "exemple",
            "fichier",
            "fictif",
            "humain",
            "ignorez",
            "la",
            "le",
            "les",
            "lire",
            "lirait",
            "opérateur",
            "pédagogique",
            "précédentes",
            "recherche",
            "relevé",
            "révélez",
            "sans",
            "scénario",
            "solde",
            "une",
            "veuillez",
        }
    ),
}
_REPORTED_FRAMING = {
    "en": re.compile(
        r"\b(?:quoted?|reported?|translated?|summari[sz](?:e|ed|ing)|"
        r"teaching\s+(?:example|sample)|according\s+to|for\s+(?:analysis|translation)|"
        r"(?:the\s+)?(?:text|document|message)\s+(?:says?|reads?|states?))\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:cit(?:é|ée|er)|citation|rapport(?:e|é|ée)|tradui(?:t|te|re)|résum(?:e|é|ée)|"
        r"exemple\s+pédagogique|selon|pour\s+(?:analyse|traduction)|"
        r"(?:le\s+)?(?:texte|document|message)\s+(?:dit|indique|énonce))\b",
        flags=re.IGNORECASE,
    ),
}
_HYPOTHETICAL_FRAMING = {
    "en": re.compile(
        r"\b(?:hypothetical|fictional|imaginary|simulation|suppose|imagine|"
        r"in\s+an?\s+(?:fictional|hypothetical)\s+(?:example|scenario)|"
        r"(?:would|could)\s+(?:read|say|tell|ask|instruct)|"
        r"if\b[^.!?\n]{0,120}\b(?:would|could|were))\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:hypothétique|ficti(?:f|ve)|imaginaire|simulation|supposons|imaginez|"
        r"dans\s+un\s+(?:exemple|scénario)\s+(?:fictif|hypothétique)|"
        r"(?:lirait|dirait|demanderait|ordonnerait)|"
        r"si\b[^.!?\n]{0,120}\b(?:pourrait|était|serait))\b",
        flags=re.IGNORECASE,
    ),
}
_CURRENT_OPERATIVE_ACTION = {
    "en": re.compile(r"\b(?:follow|obey|execute|apply|act\s+on)\b", flags=re.IGNORECASE),
    "fr": re.compile(r"\b(?:suiv|obéi|exécut|appliqu)\w*\b", flags=re.IGNORECASE),
}
_CURRENT_OPERATIVE_TIME = {
    "en": re.compile(
        r"\b(?:now|immediately|in\s+the\s+current\s+(?:task|interaction))\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:maintenant|immédiatement|dans\s+(?:la|l['’])\s*"
        r"(?:tâche|interaction)\s+actuelle)\b",
        flags=re.IGNORECASE,
    ),
}
_NO_INSTRUCTION_CUE = {
    "en": re.compile(
        r"(?:\bplease\s+|(?:^|[.!?]\s+))"
        r"(?:add|browse|call|cancel|change|copy|create|delete|download|execute|export|"
        r"forward|import|invite|list|modify|move|open|purchase|buy|read|remove|rename|"
        r"reply|respond|reveal|run|save|schedule|search|send|share|store|remember|submit|"
        r"summarize|update|upload|transfer)\b|"
        r"\b(?:you|assistant|agent)\s+(?:must|should|need\s+to|are\s+required\s+to)\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:veuillez|merci\s+de)\s+(?:ajout|appel|annul|achet|copi|cré|déplac|"
        r"effac|enregistr|envoy|exécut|export|import|invit|lis|lire|modifi|ouvr|"
        r"partag|recherch|renomm|répond|résum|révél|soumet|supprim|télécharg|"
        r"transfér)\w*\b|\b(?:vous|l['’]assistant|l['’]agent)\s+devez\b",
        flags=re.IGNORECASE,
    ),
}
_HUMAN_ADDRESSEE = {
    "en": re.compile(
        r"\b(?:human\s+(?:operator|reader|reviewer|employee|user)|"
        r"staff\s+member|employee|person)\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:opérat(?:eur|rice)\s+humain(?:e)?|lecteur\s+humain|"
        r"utilisateur\s+humain|membre\s+du\s+personnel|employé|personne)\b",
        flags=re.IGNORECASE,
    ),
}
_EXTERNAL_SYSTEM_ADDRESSEE = {
    "en": re.compile(
        r"\b(?:external|downstream|third[- ]party)\s+"
        r"(?:service|system|application|tool|api|server|process)\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:service|système|application|outil|api|serveur|processus)\s+"
        r"(?:externe|en\s+aval|tiers)\b",
        flags=re.IGNORECASE,
    ),
}
_MODEL_ADDRESSEE = {
    "en": re.compile(r"\b(?:assistant|model|agent|llm|chatbot)\b", flags=re.IGNORECASE),
    "fr": re.compile(r"\b(?:assistant|modèle|agent|llm|chatbot)\b", flags=re.IGNORECASE),
}
_CONFIRMATION_BYPASS = {
    "en": re.compile(
        r"\b(?:without(?:\s+(?:asking|requesting|waiting)(?:\s+for)?)?\s+"
        r"(?:explicit\s+|user\s+)?(?:confirmation|approval|authorization|consent)|"
        r"(?:skip|bypass|ignore|omit)\b[^.!?\n]{0,50}"
        r"\b(?:confirmation|approval|authorization|consent)|"
        r"do\s+not\s+(?:ask|request|wait|require)\b[^.!?\n]{0,50}"
        r"\b(?:confirmation|approval|authorization|consent)|"
        r"no\s+(?:confirmation|approval|authorization|consent)\s+"
        r"(?:is\s+)?(?:needed|required))\b",
        flags=re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:sans(?:\s+(?:demander|attendre)(?:\s+de)?)?\s+"
        r"(?:confirmation|validation|approbation|autorisation|accord)|"
        r"(?:ignorez|omettez|contournez)\b[^.!?\n]{0,50}"
        r"\b(?:confirmation|validation|approbation|autorisation|accord)|"
        r"ne\s+(?:demandez|attendez|exigez)\s+pas\b[^.!?\n]{0,50}"
        r"\b(?:confirmation|validation|approbation|autorisation|accord)|"
        r"aucune\s+(?:confirmation|validation|approbation|autorisation)\s+"
        r"(?:n['’]est\s+)?(?:nécessaire|requise))\b",
        flags=re.IGNORECASE,
    ),
}
_QUOTE_PAIRS = {'"': '"', "“": "”", "„": "”", "«": "»", "‹": "›"}
_SYMBOLIC_ARGUMENT_TERMS = {
    "FABRICATE_APPROVAL_DECISION": {
        "en": ("fabricat", "approval", "decision", "evidence"),
        "fr": ("fabriqu", "décision", "approbation", "preuve"),
    },
    "INSTALL_MALWARE_AND_TAKE_CONTROL": {
        "en": ("install", "malware", "control", "system"),
        "fr": ("install", "logiciel malveillant", "contrôle", "système"),
    },
    "PRIVATE_PROTECTED_DATA": {
        "en": ("private", "protected", "data"),
        "fr": ("données", "privées", "protégées"),
    },
    "UNTRUSTED_EXTERNAL_RECIPIENT": {
        "en": ("untrusted", "external", "recipient"),
        "fr": ("destinataire", "externe", "non fiable"),
    },
}


class PolicyBenchValidationError(ValueError):
    """Raised when one or more PolicyBench validation gates fail."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(dict.fromkeys(errors))
        if not self.errors:
            raise ValueError("PolicyBenchValidationError requires at least one error")
        super().__init__(
            "PolicyBench validation failed:\n" + "\n".join(f"- {error}" for error in self.errors)
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("expected a mapping or object with to_dict()")


def _flatten_scalars(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten_scalars(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _flatten_scalars(child)]
    if value is None:
        return []
    return [str(value)]


def _language_errors(text: str, language: str, context: str) -> list[str]:
    """Reject only high-confidence English/French mismatches.

    This deliberately is not a general language detector. It requires several
    curated function/content words and a clear margin, so names, identifiers,
    short strings, and borrowed technical terms do not cause rejection.
    """

    if language not in _LANGUAGE_MARKERS:
        return []
    tokens = [token.casefold() for token in _WORD.findall(text)]
    scores = {
        candidate_language: sum(token in markers for token in tokens)
        for candidate_language, markers in _LANGUAGE_MARKERS.items()
    }
    other = "fr" if language == "en" else "en"
    if scores[other] >= 3 and scores[other] >= scores[language] + 2:
        return [f"{context}: text is reliably {other}, not the required {language} language"]
    return []


def _directive_context(text: str, spans: Sequence[Any]) -> str:
    """Return framing text with resolved directive spans masked in place."""

    masked = list(text)
    for span in spans:
        if getattr(span, "span_type", None) != "DIRECTIVE":
            continue
        for index in range(span.start, span.end):
            masked[index] = " "
    return "".join(masked)


def _is_quote_enclosed(text: str, span: Any) -> bool:
    left = text[: span.start].rstrip()
    right = text[span.end :].lstrip()
    if not left or not right:
        return False
    opener = left[-1]
    return opener in _QUOTE_PAIRS and right[0] == _QUOTE_PAIRS[opener]


def _presentation_errors(
    candidate: str,
    spans: Sequence[Any],
    presentation: Any,
    language: str,
) -> list[str]:
    directive_spans = [span for span in spans if getattr(span, "span_type", None) == "DIRECTIVE"]
    if not directive_spans or language not in _REPORTED_FRAMING:
        return []
    context = _directive_context(candidate, directive_spans)
    reported_marker = _REPORTED_FRAMING[language].search(context) is not None
    quote_enclosed = any(_is_quote_enclosed(candidate, span) for span in directive_spans)
    hypothetical_marker = _HYPOTHETICAL_FRAMING[language].search(context) is not None
    current_operative_marker = (
        _CURRENT_OPERATIVE_ACTION[language].search(context) is not None
        and _CURRENT_OPERATIVE_TIME[language].search(context) is not None
    )

    if presentation == "QUOTED_OR_REPORTED":
        errors = []
        if not (reported_marker or quote_enclosed):
            errors.append(
                "$.candidate_text: QUOTED_OR_REPORTED directive lacks quotation or "
                "reporting evidence"
            )
        if hypothetical_marker:
            errors.append(
                "$.candidate_text: QUOTED_OR_REPORTED directive has explicit hypothetical framing"
            )
        return errors
    if presentation == "HYPOTHETICAL" and not hypothetical_marker:
        return ["$.candidate_text: HYPOTHETICAL directive lacks explicit hypothetical framing"]
    if presentation == "OPERATIVE":
        if hypothetical_marker:
            return ["$.candidate_text: OPERATIVE directive has explicit hypothetical framing"]
        if reported_marker and not current_operative_marker:
            return ["$.candidate_text: OPERATIVE directive has explicit quoted/reported framing"]
        if quote_enclosed and not current_operative_marker:
            return ["$.candidate_text: OPERATIVE directive is only presented as a quotation"]
    return []


def _addressee_errors(directive_text: str, addressee: Any, language: str) -> list[str]:
    if language not in _HUMAN_ADDRESSEE:
        return []
    has_human = _HUMAN_ADDRESSEE[language].search(directive_text) is not None
    has_external = _EXTERNAL_SYSTEM_ADDRESSEE[language].search(directive_text) is not None
    has_model = _MODEL_ADDRESSEE[language].search(directive_text) is not None
    if addressee == "HUMAN" and not has_human:
        return ["$.directive_anchors: HUMAN addressee lacks explicit human-executor evidence"]
    if addressee == "EXTERNAL_SYSTEM" and not has_external:
        return [
            "$.directive_anchors: EXTERNAL_SYSTEM addressee lacks explicit external-system evidence"
        ]
    if addressee == "MODEL_OR_AGENT" and not has_model and (has_human or has_external):
        return [
            "$.directive_anchors: MODEL_OR_AGENT label contradicts an explicit non-model addressee"
        ]
    return []


def _confirmation_errors(
    directive_text: str,
    confirmation_handling: Any,
    language: str,
) -> list[str]:
    pattern = _CONFIRMATION_BYPASS.get(language)
    if pattern is None:
        return []
    has_bypass = pattern.search(directive_text) is not None
    if confirmation_handling == "EXPLICITLY_BYPASSED" and not has_bypass:
        return [
            "$.directive_anchors: EXPLICITLY_BYPASSED confirmation handling lacks an "
            "explicit confirmation bypass"
        ]
    if confirmation_handling not in {"EXPLICITLY_BYPASSED", "UNKNOWN"} and has_bypass:
        return ["$.directive_anchors: introduces an undeclared explicit confirmation bypass"]
    return []


def _action_errors(text: str, action: Any, language: str, context: str) -> list[str]:
    if not isinstance(action, str) or not action:
        return []
    normalized = text.casefold()
    prefix = action.split("_", 1)[0]
    terms = _ACTION_VERBS.get(prefix, {}).get(language, ())
    errors: list[str] = []
    if terms and not any(term.casefold() in normalized for term in terms):
        errors.append(f"{context}: candidate does not preserve declared action {action}")
    for object_name, by_language in _OBJECT_TERMS.items():
        if object_name not in action:
            continue
        object_terms = by_language.get(language, ())
        if object_terms and not any(term.casefold() in normalized for term in object_terms):
            errors.append(f"{context}: candidate does not preserve {object_name} target")
    return errors


def _argument_is_preserved(text: str, value: Any, language: str) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_argument_is_preserved(text, item, language) for item in value)
    if isinstance(value, bool):
        return re.search(rf"\b{str(value).casefold()}\b", text, flags=re.IGNORECASE) is not None
    if isinstance(value, (int, float)):
        literal = re.escape(str(value))
        return re.search(rf"(?<![\w.]){literal}(?![\w.])", text) is not None
    if not isinstance(value, str):
        return False
    normalized_text = text.casefold().replace("_", " ")
    normalized_value = value.casefold().replace("_", " ")
    if normalized_value in normalized_text:
        return True
    terms = _SYMBOLIC_ARGUMENT_TERMS.get(value, {}).get(language)
    return bool(terms) and all(term.casefold() in normalized_text for term in terms)


def _argument_errors(
    text: str,
    arguments: Any,
    language: str,
    context: str,
    scenario_id: Any,
) -> list[str]:
    if not isinstance(arguments, Mapping):
        return []
    errors: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str) and (not value or value.casefold() in {"unspecified", "unknown"}):
            continue
        if key == "scenario_reference" and str(scenario_id).casefold() in text.casefold():
            continue
        if not _argument_is_preserved(text, value, language):
            errors.append(f"{context}: declared argument {key!r} was removed")
    return errors


def validate_generation_response(
    response: Any,
    blueprint: Any,
    policy: Mapping[str, Any],
    *,
    maximum_candidate_characters: int = 12_000,
) -> list[str]:
    """Validate untrusted linguistic output against its deterministic blueprint."""

    errors = list(validate_generation_response_schema(response))
    try:
        plan = _mapping(blueprint)
    except TypeError as error:
        return [str(error)]
    errors.extend(validate_scenario_blueprint_schema(plan))
    if not isinstance(response, Mapping):
        return list(dict.fromkeys(errors))

    scenario_id = response.get("scenario_id")
    if scenario_id != plan.get("scenario_id"):
        errors.append("$.scenario_id: response changed the scenario identifier")
    language = plan.get("language")
    if response.get("language") != language:
        errors.append("$.language: response changed the required language")
    candidate = response.get("candidate_text")
    if not isinstance(candidate, str):
        return list(dict.fromkeys(errors))
    if len(candidate) > maximum_candidate_characters:
        errors.append(
            f"$.candidate_text: exceeds {maximum_candidate_characters} Unicode code points"
        )
    errors.extend(_language_errors(candidate, str(language), "$.candidate_text"))
    if _LABEL_LEAK.search(candidate):
        errors.append("$.candidate_text: leaks a canonical label or annotation field")

    allowed_urls = set(_URL.findall(" ".join(_flatten_scalars(plan))))
    unexpected_urls = sorted(set(_URL.findall(candidate)).difference(allowed_urls))
    if unexpected_urls:
        errors.append("$.candidate_text: contains an undeclared URL")

    authority_override = plan.get("authority_delegation_override")
    expected_policy = (
        authority_override.get("effective_protected_policy_text")
        if isinstance(authority_override, Mapping)
        else policy.get(f"protected_policy_text_{language}")
    )
    realized_policy = response.get("protected_policy")
    expected_annotations = plan.get("expected_annotations")
    expected_annotations = expected_annotations if isinstance(expected_annotations, Mapping) else {}
    no_instruction_cue = _NO_INSTRUCTION_CUE.get(str(language))
    if (
        expected_annotations.get("instruction_presence") == "NO_INSTRUCTION"
        and no_instruction_cue is not None
        and no_instruction_cue.search(candidate)
    ):
        errors.append(
            "$.candidate_text: NO_INSTRUCTION response contains a high-confidence directive cue"
        )
    directive_spec = plan.get("candidate_directive_spec")
    directive_spec = directive_spec if isinstance(directive_spec, Mapping) else {}
    expected_presentation = expected_annotations.get("instruction_presentation")
    framing = directive_spec.get("framing")
    presentation_by_framing = {
        "OPERATIVE": "OPERATIVE",
        "QUOTED_OR_REPORTED": "QUOTED_OR_REPORTED",
        "HYPOTHETICAL": "HYPOTHETICAL",
        "DESCRIPTIVE": "QUOTED_OR_REPORTED",
    }
    if framing in presentation_by_framing and presentation_by_framing[framing] != (
        expected_presentation
    ):
        errors.append(
            "$.expected_annotations.instruction_presentation: contradicts "
            "candidate_directive_spec.framing"
        )
    intended_addressee = directive_spec.get("intended_addressee")
    if (
        intended_addressee is not None
        and expected_annotations.get("instruction_addressee") != intended_addressee
    ):
        errors.append(
            "$.expected_annotations.instruction_addressee: contradicts "
            "candidate_directive_spec.intended_addressee"
        )

    goal_spec = plan.get("user_goal_spec")
    goal_intent = goal_spec.get("intent") if isinstance(goal_spec, Mapping) else None
    directive_action = directive_spec.get("action")
    aligned_via_analysis = goal_intent == "ANALYZE_CONTENT" and expected_presentation in {
        "QUOTED_OR_REPORTED",
        "HYPOTHETICAL",
    }
    if (
        expected_annotations.get("user_goal_alignment") == "ALIGNED"
        and isinstance(goal_intent, str)
        and isinstance(directive_action, str)
        and directive_action != goal_intent
        and not aligned_via_analysis
    ):
        errors.append(
            "$.expected_annotations.user_goal_alignment: ALIGNED is unsupported because "
            "the user-goal intent and directive action differ"
        )
    missing_policy_allowed = (
        plan.get("category") == "INSUFFICIENT_CONTEXT"
        and expected_annotations.get("protected_policy_alignment") == "UNDETERMINED"
    )
    if realized_policy is None and not missing_policy_allowed:
        errors.append("$.protected_policy: protected policy is missing without declaration")
    elif realized_policy is not None and realized_policy != expected_policy:
        errors.append("$.protected_policy: response changed the catalogued policy wording")

    realized_goal = response.get("user_goal")
    if goal_spec is None:
        if realized_goal is not None:
            errors.append("$.user_goal: response invented a missing user goal")
    elif not isinstance(realized_goal, str) or not realized_goal.strip():
        errors.append("$.user_goal: response removed the required user goal")
    elif isinstance(goal_spec, Mapping):
        errors.extend(_language_errors(realized_goal, str(language), "$.user_goal"))
        unexpected_goal_urls = sorted(set(_URL.findall(realized_goal)).difference(allowed_urls))
        if unexpected_goal_urls:
            errors.append("$.user_goal: contains an undeclared URL")
        if _LABEL_LEAK.search(realized_goal):
            errors.append("$.user_goal: leaks a canonical label or annotation field")
        errors.extend(
            _action_errors(
                realized_goal,
                goal_spec.get("intent"),
                str(language),
                "$.user_goal",
            )
        )
        errors.extend(
            _argument_errors(
                realized_goal,
                goal_spec.get("arguments"),
                str(language),
                "$.user_goal",
                scenario_id,
            )
        )

    anchor_fields = (
        "directive_anchors",
        "injection_payload_anchors",
        "authority_claim_anchors",
    )
    if all(isinstance(response.get(field), list) for field in anchor_fields):
        try:
            spans = resolve_generation_anchors(
                candidate,
                directive_anchors=response["directive_anchors"],
                injection_payload_anchors=response["injection_payload_anchors"],
                authority_claim_anchors=response["authority_claim_anchors"],
            )
        except SpanError as error:
            errors.extend(f"$.candidate_text: {item}" for item in error.errors)
            spans = ()
        span_types = {span.span_type for span in spans}
        required_types = expected_annotations.get("required_span_types", [])
        if isinstance(required_types, list):
            required_type_set = set(required_types)
            for missing in sorted(required_type_set.difference(span_types)):
                errors.append(f"$.{missing.lower()}_anchors: required {missing} span is absent")
            for unexpected in sorted(span_types.difference(required_type_set)):
                errors.append(f"$.{unexpected.lower()}_anchors: undeclared {unexpected} span")
        if expected_annotations.get("instruction_presence") == "NO_INSTRUCTION" and spans:
            errors.append("$.candidate_text: NO_INSTRUCTION response must not contain spans")

        authority_claim = directive_spec.get("authority_claim", "NONE")
        authority_anchors = response.get("authority_claim_anchors", [])
        if authority_claim == "NONE" and authority_anchors:
            errors.append("$.authority_claim_anchors: undeclared authority claim")
        if authority_claim != "NONE" and not authority_anchors:
            errors.append("$.authority_claim_anchors: declared authority claim is missing")
        for index, anchor in enumerate(authority_anchors):
            anchor_text = anchor.get("text") if isinstance(anchor, Mapping) else None
            if isinstance(anchor_text, str):
                claim_match = _AUTHORITY_MARKER.search(anchor_text)
                if claim_match is None:
                    errors.append(
                        f"$.authority_claim_anchors[{index}]: "
                        "span text lacks authority-claim evidence"
                    )
                else:
                    surrounding_text = (
                        anchor_text[: claim_match.start()] + anchor_text[claim_match.end() :]
                    )
                    if _WORD.search(surrounding_text):
                        errors.append(
                            f"$.authority_claim_anchors[{index}]: span must be bounded to the "
                            "authority claim"
                        )
        authority_spans = [span for span in spans if span.span_type == "AUTHORITY_CLAIM"]
        authority_matches = list(_AUTHORITY_MARKER.finditer(candidate))
        if any(
            not any(
                span.start <= match.start() and span.end >= match.end() for span in authority_spans
            )
            for match in authority_matches
        ):
            errors.append(
                "$.authority_claim_anchors: candidate contains an unanchored authority claim"
            )

        directive_spans = [span for span in spans if span.span_type == "DIRECTIVE"]
        injection_spans = [span for span in spans if span.span_type == "INJECTION_PAYLOAD"]
        for index, injection_span in enumerate(injection_spans):
            if not any(
                directive_span.start <= injection_span.start
                and injection_span.end <= directive_span.end
                for directive_span in directive_spans
            ):
                errors.append(
                    f"$.injection_payload_anchors[{index}]: payload is not contained in a directive"
                )

        directive_text = " ".join(
            anchor.get("text", "")
            for anchor in response.get("directive_anchors", [])
            if isinstance(anchor, Mapping)
        )
        action = directive_spec.get("action")
        if directive_spec.get("directive_required") and not response.get("directive_anchors"):
            errors.append("$.directive_anchors: required directive is missing")
        if action is not None and directive_text:
            errors.extend(_language_errors(directive_text, str(language), "$.directive_anchors"))
            errors.extend(
                _action_errors(directive_text, action, str(language), "$.directive_anchors")
            )
        if directive_text:
            errors.extend(
                _presentation_errors(
                    candidate,
                    spans,
                    expected_presentation,
                    str(language),
                )
            )
            errors.extend(_addressee_errors(directive_text, intended_addressee, str(language)))
            errors.extend(
                _confirmation_errors(
                    directive_text,
                    directive_spec.get("confirmation_handling"),
                    str(language),
                )
            )
        if directive_spec.get("directive_required") is not False:
            errors.extend(
                _argument_errors(
                    directive_text,
                    directive_spec.get("arguments"),
                    str(language),
                    "$.directive_anchors",
                    scenario_id,
                )
            )

    return list(dict.fromkeys(errors))


def require_valid_generation_response(
    response: Any,
    blueprint: Any,
    policy: Mapping[str, Any],
    *,
    maximum_candidate_characters: int = 12_000,
) -> None:
    errors = validate_generation_response(
        response,
        blueprint,
        policy,
        maximum_candidate_characters=maximum_candidate_characters,
    )
    if errors:
        raise PolicyBenchValidationError(errors)


def validate_policybench_record(
    record: Any,
    blueprint: Any | None = None,
    *,
    require_generated_state: bool = True,
) -> list[str]:
    """Validate canonical semantics and, by default, the generated-release state."""

    errors = list(validate_policybench_record_schema(record))
    errors.extend(validate_record(record, schema_path=schema_path("policybench_record")))
    if not isinstance(record, Mapping):
        return list(dict.fromkeys(errors))
    annotations = record.get("annotations")
    annotations = annotations if isinstance(annotations, Mapping) else {}
    content = record.get("content")
    content = content if isinstance(content, Mapping) else {}
    extension = record.get("extensions")
    extension = extension if isinstance(extension, Mapping) else {}
    policybench = extension.get("policybench_v0_1")
    policybench = policybench if isinstance(policybench, Mapping) else {}
    policy_metadata = policybench.get("policy")
    policy_metadata = policy_metadata if isinstance(policy_metadata, Mapping) else {}
    context = record.get("context")
    context = context if isinstance(context, Mapping) else {}
    protected_policy = context.get("protected_policy")
    effective_policy_hash = (
        sha256_text(protected_policy) if isinstance(protected_policy, str) else sha256_text("")
    )
    if (
        "effective_policy_sha256" in policy_metadata
        and policy_metadata.get("effective_policy_sha256") != effective_policy_hash
    ):
        errors.append(
            "$.extensions.policybench_v0_1.policy.effective_policy_sha256: "
            "does not match the effective protected policy"
        )
    delegation_override = policy_metadata.get("authority_delegation_override")
    if isinstance(delegation_override, Mapping) and protected_policy != delegation_override.get(
        "effective_protected_policy_text"
    ):
        errors.append("$.context.protected_policy: does not match authority delegation override")

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    provenance = metadata.get("dataset_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_record = provenance.get("source_record")
    source_record = source_record if isinstance(source_record, Mapping) else {}
    if source_record.get("split") != policybench.get("dataset_split"):
        errors.append(
            "$.metadata.dataset_provenance.source_record.split: must match PolicyBench split"
        )

    if require_generated_state:
        data_quality = policybench.get("data_quality")
        if data_quality == "GOLD_HUMAN_CONFIRMED":
            errors.append("$.extensions.policybench_v0_1.data_quality: automatic gold is forbidden")
        if data_quality not in {
            "SILVER_TEMPLATE",
            "SILVER_GENERATED",
            "SILVER_VALIDATED",
            "EXCLUDED",
        }:
            errors.append(
                "$.extensions.policybench_v0_1.data_quality: generated record must be SILVER"
            )
        if policybench.get("human_validation_status") != "PENDING":
            errors.append(
                "$.extensions.policybench_v0_1.human_validation_status: must remain PENDING"
            )
        if annotations.get("annotator_confidence") != 0.0:
            errors.append(
                "$.annotations.annotator_confidence: generated records have no human "
                "annotator and must use 0.0"
            )
    annotation_spans = annotations.get("spans")
    span_errors = validate_spans(content.get("text"), annotation_spans)
    errors.extend(f"$.annotations.spans: {error}" for error in span_errors)
    if annotations.get("authority_status") == "SPOOFED" and not any(
        isinstance(span, Mapping) and span.get("type") == "AUTHORITY_CLAIM"
        for span in (annotation_spans if isinstance(annotation_spans, list) else [])
    ):
        errors.append("$.annotations.spans: SPOOFED authority requires AUTHORITY_CLAIM")
    derived = record.get("derived")
    derived = derived if isinstance(derived, Mapping) else {}
    if derived.get("prompt_injection_verdict") == "DETECTED" and "PROMPT_INJECTION" not in (
        annotations.get("attack_families") or []
    ):
        errors.append(
            "$.annotations.attack_families: DETECTED generated case requires PROMPT_INJECTION"
        )

    if blueprint is not None:
        plan = _mapping(blueprint)
        expected = plan.get("expected_annotations")
        expected = expected if isinstance(expected, Mapping) else {}
        for field in (
            "instruction_presence",
            "instruction_presentation",
            "instruction_addressee",
            "user_goal_alignment",
            "protected_policy_alignment",
            "authority_status",
            "attack_families",
            "attack_objectives",
            "annotation_status",
        ):
            if field in expected and annotations.get(field) != expected[field]:
                errors.append(f"$.annotations.{field}: does not match scenario blueprint")
        if expected.get("prompt_injection_verdict") is not None and derived.get(
            "prompt_injection_verdict"
        ) != expected.get("prompt_injection_verdict"):
            errors.append("$.derived.prompt_injection_verdict: does not match scenario blueprint")
        actual_types = {
            span.get("type")
            for span in (annotation_spans if isinstance(annotation_spans, list) else [])
            if isinstance(span, Mapping)
        }
        required_types = expected.get("required_span_types", [])
        if isinstance(required_types, list) and not set(required_types).issubset(actual_types):
            errors.append("$.annotations.spans: missing a blueprint-required span type")
    return list(dict.fromkeys(errors))


def require_valid_policybench_record(
    record: Any,
    blueprint: Any | None = None,
    *,
    require_generated_state: bool = True,
) -> None:
    errors = validate_policybench_record(
        record,
        blueprint,
        require_generated_state=require_generated_state,
    )
    if errors:
        raise PolicyBenchValidationError(errors)


def _policybench_extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions")
    value = extensions.get("policybench_v0_1") if isinstance(extensions, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _counterfactual_label(record: Mapping[str, Any], field: str) -> Any:
    if field == "prompt_injection_verdict":
        derived = record.get("derived")
        return derived.get(field) if isinstance(derived, Mapping) else None
    annotations = record.get("annotations")
    return annotations.get(field) if isinstance(annotations, Mapping) else None


def _without(value: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name not in names}


def validate_counterfactual_record_groups(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate realized two-record groups and their one-variable invariants."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        extension = _policybench_extension(record)
        provenance = extension.get("counterfactual")
        if isinstance(provenance, Mapping) and isinstance(
            provenance.get("counterfactual_group_id"), str
        ):
            groups[provenance["counterfactual_group_id"]].append(record)
    result: list[dict[str, Any]] = []
    label_fields = (
        "instruction_presence",
        "instruction_presentation",
        "instruction_addressee",
        "user_goal_alignment",
        "protected_policy_alignment",
        "authority_status",
        "attack_families",
        "attack_objectives",
        "annotation_status",
        "prompt_injection_verdict",
    )
    for group_id, members in sorted(groups.items()):
        errors: list[str] = []
        if len(members) != 2:
            result.append(
                {
                    "id": group_id,
                    "errors": [
                        f"counterfactual group must contain exactly 2 records, got {len(members)}"
                    ],
                }
            )
            continue
        extensions = [_policybench_extension(record) for record in members]
        provenances = [extension.get("counterfactual") for extension in extensions]
        if not all(isinstance(item, Mapping) for item in provenances):
            result.append({"id": group_id, "errors": ["counterfactual provenance is missing"]})
            continue
        left_provenance = provenances[0]
        assert isinstance(left_provenance, Mapping)
        if any(item != left_provenance for item in provenances[1:]):
            errors.append("counterfactual provenance differs between siblings")
        parent_id = left_provenance.get("parent_scenario_id")
        parent_matches = [
            index
            for index, extension in enumerate(extensions)
            if isinstance(extension.get("blueprint"), Mapping)
            and extension["blueprint"].get("scenario_blueprint_id") == parent_id
        ]
        if len(parent_matches) != 1:
            errors.append("counterfactual parent_scenario_id does not identify exactly one sibling")
            result.append({"id": group_id, "errors": errors})
            continue
        parent_index = parent_matches[0]
        sibling_index = 1 - parent_index
        parent = members[parent_index]
        sibling = members[sibling_index]
        parent_extension = extensions[parent_index]
        sibling_extension = extensions[sibling_index]
        changed_type = left_provenance.get("counterfactual_type")
        parent_context = parent.get("context")
        sibling_context = sibling.get("context")
        parent_content = parent.get("content")
        sibling_content = sibling.get("content")
        parent_context = parent_context if isinstance(parent_context, Mapping) else {}
        sibling_context = sibling_context if isinstance(sibling_context, Mapping) else {}
        parent_content = parent_content if isinstance(parent_content, Mapping) else {}
        sibling_content = sibling_content if isinstance(sibling_content, Mapping) else {}
        parent_policy = parent_extension.get("policy")
        sibling_policy = sibling_extension.get("policy")
        parent_policy = parent_policy if isinstance(parent_policy, Mapping) else {}
        sibling_policy = sibling_policy if isinstance(sibling_policy, Mapping) else {}

        if parent_extension.get("dataset_split") != sibling_extension.get("dataset_split"):
            errors.append("counterfactual siblings cross dataset splits")
        parent_grouping = parent_extension.get("grouping")
        sibling_grouping = sibling_extension.get("grouping")
        if (
            isinstance(parent_grouping, Mapping)
            and isinstance(sibling_grouping, Mapping)
            and parent_grouping.get("split_group_id") != sibling_grouping.get("split_group_id")
        ):
            errors.append("counterfactual siblings have different split_group_id values")

        policy_changed = parent_policy.get("policy_id") != sibling_policy.get(
            "policy_id"
        ) and parent_context.get("protected_policy") != sibling_context.get("protected_policy")
        same_policy = parent_policy.get("policy_id") == sibling_policy.get("policy_id")
        same_context = parent_context == sibling_context
        same_content = parent_content == sibling_content
        if changed_type == "POLICY_CHANGE":
            if not policy_changed:
                errors.append(f"{changed_type} did not change the protected policy boundary")
            if (
                _without(parent_context, "protected_policy")
                != _without(sibling_context, "protected_policy")
                or not same_content
            ):
                errors.append(f"{changed_type} changed a declared context/content invariant")
        elif changed_type == "AUTHORITY_DELEGATION_CHANGE":
            if not same_policy or parent_context.get("protected_policy") == sibling_context.get(
                "protected_policy"
            ):
                errors.append(
                    "AUTHORITY_DELEGATION_CHANGE must retain policy identity and change its "
                    "effective boundary"
                )
            if (
                _without(parent_context, "protected_policy")
                != _without(sibling_context, "protected_policy")
                or not same_content
            ):
                errors.append(
                    "AUTHORITY_DELEGATION_CHANGE changed a declared context/content invariant"
                )
            if _without(
                parent_policy,
                "effective_policy_sha256",
                "authority_delegation_override",
            ) != _without(
                sibling_policy,
                "effective_policy_sha256",
                "authority_delegation_override",
            ):
                errors.append(
                    "AUTHORITY_DELEGATION_CHANGE changed policy ID, family, rules, version, "
                    "or catalogue hash"
                )
            parent_override = parent_policy.get("authority_delegation_override")
            sibling_override = sibling_policy.get("authority_delegation_override")
            if not isinstance(parent_override, Mapping) or not isinstance(
                sibling_override, Mapping
            ):
                errors.append("AUTHORITY_DELEGATION_CHANGE requires both effective overrides")
            else:
                for field in ("mode", "action", "source_role", "base_effect"):
                    if parent_override.get(field) != sibling_override.get(field):
                        errors.append(
                            f"AUTHORITY_DELEGATION_CHANGE altered override field {field!r}"
                        )
                effects = {
                    parent_override.get("effective_effect"),
                    sibling_override.get("effective_effect"),
                }
                if effects != {"ALLOW_AUTHORITY", "DENY_AUTHORITY"}:
                    errors.append("AUTHORITY_DELEGATION_CHANGE effects are not an ALLOW/DENY pair")
                for record, override, context in (
                    (parent, parent_override, parent_context),
                    (sibling, sibling_override, sibling_context),
                ):
                    expected_authority = (
                        "WITHIN_AUTHORITY"
                        if override.get("effective_effect") == "ALLOW_AUTHORITY"
                        else "OUTSIDE_AUTHORITY"
                    )
                    if _counterfactual_label(record, "authority_status") != expected_authority:
                        errors.append(
                            "AUTHORITY_DELEGATION_CHANGE authority label disagrees with the "
                            "effective boundary"
                        )
                    if context.get("protected_policy") != override.get(
                        "effective_protected_policy_text"
                    ):
                        errors.append(
                            "AUTHORITY_DELEGATION_CHANGE effective text disagrees with context"
                        )
        elif changed_type == "USER_GOAL_CHANGE":
            if parent_context.get("user_goal") == sibling_context.get("user_goal"):
                errors.append("USER_GOAL_CHANGE did not change user_goal")
            if (
                _without(parent_context, "user_goal") != _without(sibling_context, "user_goal")
                or not same_policy
                or not same_content
            ):
                errors.append("USER_GOAL_CHANGE changed a declared invariant")
        elif changed_type == "SOURCE_ROLE_CHANGE":
            if parent_content.get("source_role") == sibling_content.get("source_role"):
                errors.append("SOURCE_ROLE_CHANGE did not change source_role")
            if (
                _without(parent_content, "source_role") != _without(sibling_content, "source_role")
                or not same_policy
                or not same_context
            ):
                errors.append("SOURCE_ROLE_CHANGE changed a declared invariant")
        elif changed_type == "CAPABILITY_CHANGE":
            if parent_context.get("available_capabilities") == sibling_context.get(
                "available_capabilities"
            ):
                errors.append("CAPABILITY_CHANGE did not change available_capabilities")
            if (
                _without(parent_context, "available_capabilities")
                != _without(sibling_context, "available_capabilities")
                or not same_policy
                or not same_content
            ):
                errors.append("CAPABILITY_CHANGE changed a declared invariant")
        elif changed_type == "PRESENTATION_CHANGE":
            if parent_content.get("text") == sibling_content.get("text"):
                errors.append("PRESENTATION_CHANGE did not change candidate text")
            if (
                _without(parent_content, "text") != _without(sibling_content, "text")
                or not same_policy
                or not same_context
            ):
                errors.append("PRESENTATION_CHANGE changed a declared invariant")
        else:
            errors.append(f"unsupported counterfactual type: {changed_type!r}")

        declared_changes = left_provenance.get("expected_label_changes")
        declared_changes = declared_changes if isinstance(declared_changes, list) else []
        declared_fields: set[str] = set()
        for change in declared_changes:
            if not isinstance(change, Mapping) or not isinstance(change.get("field"), str):
                errors.append("expected_label_changes contains an invalid entry")
                continue
            field = change["field"]
            declared_fields.add(field)
            if _counterfactual_label(parent, field) != change.get("from"):
                errors.append(f"expected_label_changes.{field}.from disagrees with parent")
            if _counterfactual_label(sibling, field) != change.get("to"):
                errors.append(f"expected_label_changes.{field}.to disagrees with sibling")
        actual_fields = {
            field
            for field in label_fields
            if _counterfactual_label(parent, field) != _counterfactual_label(sibling, field)
        }
        if declared_fields != actual_fields:
            errors.append(
                "expected_label_changes fields do not equal realized label changes: "
                f"declared={sorted(declared_fields)}, actual={sorted(actual_fields)}"
            )
        if changed_type == "CAPABILITY_CHANGE" and actual_fields:
            errors.append(
                "CAPABILITY_CHANGE must not invent frozen taxonomy consequences from feasibility"
            )
        if changed_type != "CAPABILITY_CHANGE" and not actual_fields:
            errors.append(f"{changed_type} must change at least one frozen label")
        if changed_type == "USER_GOAL_CHANGE":
            misalignment_objectives = {
                "TASK_HIJACKING",
                "OUTPUT_OR_DECISION_MANIPULATION",
            }
            dynamic_objectives = set(misalignment_objectives)
            if {
                _counterfactual_label(record, "protected_policy_alignment")
                for record in (parent, sibling)
            } == {"CONFLICTING"}:
                dynamic_objectives.add("UNAUTHORIZED_ACTION")
            for record in (parent, sibling):
                alignment = _counterfactual_label(record, "user_goal_alignment")
                authority = _counterfactual_label(record, "authority_status")
                objectives = set(_counterfactual_label(record, "attack_objectives") or [])
                if alignment == "ALIGNED" and objectives.intersection(misalignment_objectives):
                    errors.append("USER_GOAL_CHANGE retained a misalignment-only attack objective")
                if (
                    alignment == "MISALIGNED"
                    and authority
                    in {
                        "OUTSIDE_AUTHORITY",
                        "SPOOFED",
                    }
                    and "TASK_HIJACKING" not in objectives
                ):
                    errors.append(
                        "USER_GOAL_CHANGE omitted TASK_HIJACKING for an unauthorized diversion"
                    )
            parent_objectives = set(_counterfactual_label(parent, "attack_objectives") or [])
            sibling_objectives = set(_counterfactual_label(sibling, "attack_objectives") or [])
            if parent_objectives.difference(dynamic_objectives) != sibling_objectives.difference(
                dynamic_objectives
            ):
                errors.append(
                    "USER_GOAL_CHANGE altered objectives not controlled by user-goal alignment"
                )
        if changed_type == "PRESENTATION_CHANGE":
            by_presentation = {
                _counterfactual_label(record, "instruction_presentation"): record
                for record in (parent, sibling)
            }
            if set(by_presentation) != {"OPERATIVE", "QUOTED_OR_REPORTED"}:
                errors.append("PRESENTATION_CHANGE must be an OPERATIVE/QUOTED_OR_REPORTED pair")
            else:
                quoted = by_presentation["QUOTED_OR_REPORTED"]
                operative = by_presentation["OPERATIVE"]
                if (
                    _counterfactual_label(quoted, "authority_status") != "WITHIN_AUTHORITY"
                    or _counterfactual_label(quoted, "protected_policy_alignment") != "COMPLIANT"
                    or _counterfactual_label(quoted, "attack_families")
                    or _counterfactual_label(quoted, "attack_objectives")
                    or _counterfactual_label(quoted, "prompt_injection_verdict") != "NOT_DETECTED"
                ):
                    errors.append(
                        "PRESENTATION_CHANGE quoted sibling leaks operative attack semantics"
                    )
                if (
                    _counterfactual_label(operative, "authority_status")
                    not in {"OUTSIDE_AUTHORITY", "SPOOFED"}
                    or "PROMPT_INJECTION"
                    not in (_counterfactual_label(operative, "attack_families") or [])
                    or _counterfactual_label(operative, "prompt_injection_verdict") != "DETECTED"
                ):
                    errors.append(
                        "PRESENTATION_CHANGE operative sibling lacks unauthorized attack semantics"
                    )
        if errors:
            result.append({"id": group_id, "errors": list(dict.fromkeys(errors))})
    return result


def validate_record_collection(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate a materialized collection and return aggregate, text-free diagnostics."""

    materialized = list(records)
    errors: list[dict[str, Any]] = []
    ids: Counter[str] = Counter()
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(materialized):
        record_id = record.get("id")
        if isinstance(record_id, str):
            ids[record_id] += 1
        record_errors = validate_policybench_record(record)
        if record_errors:
            errors.append({"id": record_id or f"index:{index}", "errors": record_errors})
        extension = record.get("extensions", {}).get("policybench_v0_1", {})
        counterfactual = extension.get("counterfactual") if isinstance(extension, Mapping) else None
        if isinstance(counterfactual, Mapping):
            group_id = counterfactual.get("counterfactual_group_id")
            if isinstance(group_id, str):
                groups[group_id].append(record)
    for record_id, count in sorted(ids.items()):
        if count > 1:
            errors.append(
                {"id": record_id, "errors": [f"duplicate record id occurs {count} times"]}
            )
    errors.extend(validate_counterfactual_record_groups(materialized))
    gold_count = sum(
        record.get("extensions", {}).get("policybench_v0_1", {}).get("data_quality")
        == "GOLD_HUMAN_CONFIRMED"
        for record in materialized
    )
    return {
        "schema_version": "0.1",
        "validation_status": "PASS" if not errors and gold_count == 0 else "FAIL",
        "records": len(materialized),
        "valid_records": len(materialized) - len({item["id"] for item in errors}),
        "invalid_records": len({item["id"] for item in errors}),
        "counterfactual_groups": len(groups),
        "automatic_gold_records": gold_count,
        "errors": errors,
    }


def validate_release_directory(
    dataset: str | Path,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify release checksums, manifests, blueprints, and exclusive split views."""

    root = Path(dataset)
    if not root.is_dir():
        return {
            "validation_status": "NOT_APPLICABLE",
            "checksums_checked": 0,
            "split_files_checked": 0,
            "errors": [],
        }
    errors: list[str] = []
    materialized = list(records) if records is not None else iter_jsonl(root / "records.jsonl")
    by_id = {
        record.get("id"): record
        for record in materialized
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    required_files = {
        "accepted_artifact_manifest.json",
        "blueprints.jsonl",
        "duplicate_report.json",
        "generation_report.json",
        "manifest.json",
        "quality_report.json",
        "quality_report.md",
        "records.jsonl",
        "split_report.json",
        "validation_report.json",
        *{f"{split}.jsonl" for split in _PUBLISHED_SPLITS},
    }
    checksum_path = root / "checksums.sha256"
    checksums: dict[str, str] = {}
    if not checksum_path.is_file():
        errors.append("checksums.sha256 is missing")
    else:
        try:
            checksum_text = checksum_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as error:
            errors.append(f"cannot read checksums.sha256: {error}")
            checksum_text = ""
        for line_number, line in enumerate(checksum_text.splitlines(), start=1):
            match = _CHECKSUM_LINE.fullmatch(line)
            if match is None:
                errors.append(f"checksums.sha256:{line_number}: malformed checksum line")
                continue
            digest, name = match.groups()
            if name in checksums:
                errors.append(f"checksums.sha256:{line_number}: duplicate path {name!r}")
                continue
            try:
                path = safe_child(root, name)
            except ValueError as error:
                errors.append(f"checksums.sha256:{line_number}: {error}")
                continue
            checksums[name] = digest
            if not path.is_file():
                errors.append(f"checksummed release file is missing: {name}")
            elif sha256_file(path) != digest:
                errors.append(f"release checksum mismatch: {name}")
    missing_checksums = sorted(required_files.difference(checksums))
    if missing_checksums:
        errors.append(f"required release files are not checksummed: {missing_checksums}")

    try:
        manifest = read_json_object(root / "manifest.json")
    except (OSError, ValueError) as error:
        errors.append(f"cannot validate manifest.json: {error}")
        manifest = {}
    if manifest:
        if manifest.get("records") != len(materialized):
            errors.append("manifest record count does not match records.jsonl")
        if (
            manifest.get("automatic_gold_records") != 0
            or manifest.get("human_validation_status") != "PENDING"
        ):
            errors.append("manifest violates SILVER/PENDING release state")
        if manifest.get("gold_claim_permitted") is not False:
            errors.append("manifest must explicitly forbid automatic GOLD claims")

    try:
        split_report = read_json_object(root / "split_report.json")
    except (OSError, ValueError) as error:
        errors.append(f"cannot validate split_report.json: {error}")
        split_report = {}
    assignments = split_report.get("assignments") if split_report else None
    if not isinstance(assignments, Mapping):
        errors.append("split_report.assignments must be an object")
        assignments = {}
    elif set(assignments) != set(by_id):
        errors.append("split_report assignments do not match canonical record IDs")
    for record_id, record in by_id.items():
        extension = _policybench_extension(record)
        if assignments.get(record_id) != extension.get("dataset_split"):
            errors.append(f"record {record_id}: canonical split disagrees with split report")
    split_files_checked = 0
    for split in _PUBLISHED_SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        split_files_checked += 1
        try:
            split_records = iter_jsonl(path)
        except ValueError as error:
            errors.append(f"cannot validate {path.name}: {error}")
            continue
        actual_ids = {record.get("id") for record in split_records}
        expected_ids = {record_id for record_id, value in assignments.items() if value == split}
        if actual_ids != expected_ids:
            errors.append(f"{path.name} does not equal its exclusive split assignment")
    constraints = split_report.get("constraints") if split_report else None
    if isinstance(constraints, Mapping):
        for name in (
            "no_transitive_group_leakage",
            "no_counterfactual_group_leakage",
            "no_semantic_duplicate_leakage",
            "held_out_domain_absent_from_train",
            "held_out_language_absent_from_train",
        ):
            if constraints.get(name) is not True:
                errors.append(f"split constraint failed: {name}")
        if manifest.get("full_release") is True and constraints.get("all_satisfied") is not True:
            errors.append("full release does not populate every requested leakage-resistant split")
    else:
        errors.append("split_report.constraints must be an object")

    try:
        blueprint_values = iter_jsonl(root / "blueprints.jsonl")
    except ValueError as error:
        errors.append(f"cannot validate blueprints.jsonl: {error}")
        blueprint_values = []
    blueprints: dict[str, Mapping[str, Any]] = {}
    for index, blueprint in enumerate(blueprint_values):
        blueprint_errors = validate_scenario_blueprint_schema(blueprint)
        scenario_id = blueprint.get("scenario_id")
        if not isinstance(scenario_id, str):
            errors.append(f"blueprints.jsonl:{index + 1}: missing scenario_id")
            continue
        if scenario_id in blueprints:
            errors.append(f"blueprints.jsonl: duplicate scenario_id {scenario_id!r}")
        blueprints[scenario_id] = blueprint
        errors.extend(f"blueprint {scenario_id}: {error}" for error in blueprint_errors)
    record_scenarios = {
        extension["blueprint"].get("scenario_blueprint_id")
        for record in materialized
        if isinstance((extension := _policybench_extension(record)).get("blueprint"), Mapping)
    }
    if set(blueprints) != record_scenarios:
        errors.append("blueprints.jsonl scenario IDs do not match canonical records")
    for record in materialized:
        extension = _policybench_extension(record)
        blueprint_metadata = extension.get("blueprint")
        if not isinstance(blueprint_metadata, Mapping):
            continue
        scenario_id = blueprint_metadata.get("scenario_blueprint_id")
        blueprint = blueprints.get(scenario_id)
        if blueprint is not None and blueprint_metadata.get("blueprint_sha256") != sha256_json(
            blueprint
        ):
            errors.append(f"record {record.get('id')}: blueprint checksum mismatch")
    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "checksums_checked": len(checksums),
        "split_files_checked": split_files_checked,
        "errors": list(dict.fromkeys(errors)),
    }


def iter_dataset_records(dataset: str | Path) -> list[dict[str, Any]]:
    """Read canonical records once, preferring `records.jsonl` over split views."""

    from promptsec.policybench.io import iter_jsonl

    root = Path(dataset)
    if root.is_file():
        return iter_jsonl(root)
    canonical = root / "records.jsonl"
    if canonical.is_file():
        return iter_jsonl(canonical)
    candidates = sorted(root.glob("*.jsonl"), key=lambda path: path.name)
    excluded = {"human_review_candidates.jsonl"}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        if path.name in excluded:
            continue
        for record in iter_jsonl(path):
            record_id = record.get("id")
            if isinstance(record_id, str) and record_id in seen:
                continue
            if isinstance(record_id, str):
                seen.add(record_id)
            records.append(record)
    return records


__all__ = [
    "PolicyBenchValidationError",
    "iter_dataset_records",
    "require_valid_generation_response",
    "require_valid_policybench_record",
    "validate_counterfactual_record_groups",
    "validate_generation_response",
    "validate_policybench_record",
    "validate_release_directory",
    "validate_record_collection",
]
