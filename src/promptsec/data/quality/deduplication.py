"""Deterministic exact and lexical-semantic duplicate analysis.

The module deliberately avoids learned embeddings. It first groups records by a
context-aware hash, binds variants that share an explicit source-template group, then
compares remaining representatives with a documented mixture of token,
token-bigram, and character-trigram Jaccard similarities.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Literal

from promptsec.data.hashing import sha256_json, sha256_text

DedupDecision = Literal["KEEP", "DROP_EXACT_DUPLICATE", "KEEP_VARIANT", "REVIEW"]

_ALGORITHM_VERSION = "lexical-semantic-v1"
_DECISIONS: tuple[DedupDecision, ...] = (
    "KEEP",
    "DROP_EXACT_DUPLICATE",
    "KEEP_VARIANT",
    "REVIEW",
)
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SYNONYMS = {
    "ignore": "ignore",
    "disregard": "ignore",
    "forget": "ignore",
    "previous": "previous",
    "prior": "previous",
    "earlier": "previous",
    "above": "previous",
    "instruction": "instruction",
    "instructions": "instruction",
    "direction": "instruction",
    "directions": "instruction",
    "rule": "instruction",
    "rules": "instruction",
}
_FILLER_TOKENS = frozenset({"all", "any", "the"})
_FEATURE_WEIGHTS = {
    "token_jaccard": 0.50,
    "token_bigram_jaccard": 0.25,
    "character_trigram_jaccard": 0.25,
}


class DeduplicationError(ValueError):
    """Raised when records or thresholds cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class DedupConfig:
    """Thresholds for representative-based semantic clustering.

    ``semantic_threshold`` is the minimum similarity needed to join an existing
    cluster.  A non-exact member at or above ``variant_threshold`` is retained as a
    useful variant; a member between the thresholds is sent to review.
    """

    semantic_threshold: float = 0.65
    variant_threshold: float = 0.85

    def __post_init__(self) -> None:
        for name, value in (
            ("semantic_threshold", self.semantic_threshold),
            ("variant_threshold", self.variant_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DeduplicationError(f"{name} must be a number between 0 and 1")
            if not 0.0 <= float(value) <= 1.0:
                raise DeduplicationError(f"{name} must be between 0 and 1")
        if self.variant_threshold < self.semantic_threshold:
            raise DeduplicationError(
                "variant_threshold must be greater than or equal to semantic_threshold"
            )


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Assignments and audit material produced by :func:`analyze_duplicates`."""

    assignments: dict[str, dict[str, Any]]
    kept_ids: tuple[str, ...]
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedRecord:
    record_id: str
    record: Mapping[str, Any]
    text: str
    normalized_text: str
    semantic_text: str
    semantic_tokens: tuple[str, ...]
    token_features: frozenset[str]
    bigram_features: frozenset[tuple[str, str]]
    character_features: frozenset[str]
    raw_hash: str
    normalized_hash: str
    contextual_hash: str


@dataclass(slots=True)
class _SemanticCluster:
    cluster_id: str
    representative_id: str
    representative: _PreparedRecord
    exact_representative_ids: list[str]


def normalize_text(text: str) -> str:
    """Return NFKC + casefold + collapsed-whitespace text without mutating input."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _semantic_tokens(normalized_text: str) -> tuple[str, ...]:
    tokens = [
        _SYNONYMS.get(token, token)
        for token in _TOKEN.findall(normalized_text)
        if token not in _FILLER_TOKENS
    ]

    # "instructions above" and "previous instructions" express the same ordering.
    # Canonicalizing the adjacent pair keeps the documented example together while
    # leaving the remaining word order available to the bigram feature.
    index = 0
    while index + 1 < len(tokens):
        if tokens[index] == "instruction" and tokens[index + 1] == "previous":
            tokens[index : index + 2] = ["previous", "instruction"]
        index += 1
    return tuple(tokens)


def _canonical_context(record: Mapping[str, Any], normalized_text: str) -> dict[str, Any]:
    content = _mapping_at(record, "content")
    annotations = _mapping_at(record, "annotations")
    return {
        "normalized_text": normalized_text,
        "context": record.get("context"),
        "delivery_mode": content.get("delivery_mode"),
        "source_role": content.get("source_role"),
        "content_origin": content.get("content_origin"),
        "ingestion_path": content.get("ingestion_path"),
        "modality": content.get("modality"),
        "instruction_presentation": annotations.get("instruction_presentation"),
        "instruction_addressee": annotations.get("instruction_addressee"),
        "authority_status": annotations.get("authority_status"),
    }


def _mapping_at(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise DeduplicationError(f"record {record.get('id')!r}: {key} must be an object")
    return value


def _prepare_record(record: Mapping[str, Any]) -> _PreparedRecord:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise DeduplicationError("every record must have a non-empty string id")
    content = _mapping_at(record, "content")
    text = content.get("text")
    if not isinstance(text, str):
        raise DeduplicationError(f"record {record_id!r}: content.text must be a string")
    # Validate annotations here so missing axes cannot silently collapse unrelated
    # malformed records into the same contextual hash.
    _mapping_at(record, "annotations")

    normalized_text = normalize_text(text)
    tokens = _semantic_tokens(normalized_text)
    semantic_text = " ".join(tokens)
    return _PreparedRecord(
        record_id=record_id,
        record=record,
        text=text,
        normalized_text=normalized_text,
        semantic_text=semantic_text,
        semantic_tokens=tokens,
        token_features=frozenset(tokens),
        bigram_features=frozenset(_token_bigrams(tokens)),
        character_features=frozenset(_character_ngrams(semantic_text)),
        raw_hash=sha256_text(text),
        normalized_hash=sha256_text(normalized_text),
        contextual_hash=sha256_json(_canonical_context(record, normalized_text)),
    )


def _jaccard(left: Set[Any], right: Set[Any]) -> float:
    if not left and not right:
        return 1.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    intersection_size = sum(item in larger for item in smaller)
    union_size = len(left) + len(right) - intersection_size
    return intersection_size / union_size if union_size else 0.0


def _token_bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


def _character_ngrams(text: str, size: int = 3) -> set[str]:
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _semantic_similarity(left: _PreparedRecord, right: _PreparedRecord) -> float:
    if left.normalized_hash == right.normalized_hash:
        return 1.0
    if left.semantic_text == right.semantic_text:
        return 1.0

    scores = {
        "token_jaccard": _jaccard(left.token_features, right.token_features),
        "token_bigram_jaccard": _jaccard(left.bigram_features, right.bigram_features),
        "character_trigram_jaccard": _jaccard(left.character_features, right.character_features),
    }
    return sum(scores[name] * weight for name, weight in _FEATURE_WEIGHTS.items())


def _cluster_id(representative_id: str) -> str:
    return f"semantic_{sha256_text(representative_id)[:24]}"


def _source_name(record: Mapping[str, Any]) -> str:
    provenance = _provenance(record)
    source = provenance.get("source_dataset")
    if isinstance(source, Mapping):
        for key in ("name", "id"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return "UNKNOWN"


def _provenance(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    provenance = metadata.get("dataset_provenance")
    return provenance if isinstance(provenance, Mapping) else {}


def _public_duplicate_member(record_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Return provenance identifiers that are safe for a public deduplication report.

    The canonical records and local review queues retain the complete source object.  A
    report committed as release metadata must not repeat source text or ``original_fields``;
    identifiers and cryptographic checksums are sufficient to audit the duplicate decision
    against a locally reconstructed payload.
    """

    provenance = _provenance(record)
    source_dataset = provenance.get("source_dataset")
    source_dataset = source_dataset if isinstance(source_dataset, Mapping) else {}
    source_record = provenance.get("source_record")
    source_record = source_record if isinstance(source_record, Mapping) else {}
    provenance_checksums = provenance.get("checksums")
    provenance_checksums = provenance_checksums if isinstance(provenance_checksums, Mapping) else {}

    checksums = {
        key: value
        for key, value in (
            ("raw_record_sha256", source_record.get("raw_record_sha256")),
            ("source_text_sha256", provenance_checksums.get("source_text_sha256")),
            ("canonical_text_sha256", provenance_checksums.get("canonical_text_sha256")),
        )
        if isinstance(value, str) and value
    }
    source_id = source_dataset.get("id")
    revision = source_dataset.get("revision")
    upstream_record_id = source_record.get("id")
    return {
        "id": record_id,
        "source": _source_name(record),
        "source_id": source_id if isinstance(source_id, str) and source_id else "UNKNOWN",
        "revision": revision if isinstance(revision, str) and revision else None,
        "source_record_id": (
            upstream_record_id
            if isinstance(upstream_record_id, str) and upstream_record_id
            else None
        ),
        "checksums": checksums,
    }


def _size_distribution(groups: Iterable[Sequence[Any]]) -> dict[str, int]:
    counts = Counter(len(group) for group in groups)
    return {str(size): counts[size] for size in sorted(counts)}


def _validated_semantic_group_keys(
    value: Mapping[str, str] | None,
    prepared: Mapping[str, _PreparedRecord],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DeduplicationError("semantic_group_keys must be a record-id mapping")
    unknown_ids = sorted(set(value) - set(prepared))
    if unknown_ids:
        raise DeduplicationError(f"semantic_group_keys contains unknown ids: {unknown_ids}")
    result: dict[str, str] = {}
    for record_id, group_key in value.items():
        if not isinstance(record_id, str) or not record_id:
            raise DeduplicationError("semantic_group_keys ids must be non-empty strings")
        if not isinstance(group_key, str) or not group_key:
            raise DeduplicationError(
                f"semantic_group_keys[{record_id!r}] must be a non-empty string"
            )
        result[record_id] = group_key
    return result


def analyze_duplicates(
    records: Iterable[Mapping[str, Any]],
    config: DedupConfig,
    *,
    semantic_group_keys: Mapping[str, str] | None = None,
) -> DedupResult:
    """Compute deterministic hashes, exact groups, and semantic clusters.

    Input order never affects the result: IDs, exact representatives, candidate
    clusters, report members, and distributions are all processed in lexical order.
    ``semantic_group_keys`` can bind variants generated from the same explicit
    source template or metadata group before lexical similarity is considered.
    Records are read only and no canonical text is rewritten.
    """

    if not isinstance(config, DedupConfig):
        raise DeduplicationError("config must be a DedupConfig")

    prepared: dict[str, _PreparedRecord] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise DeduplicationError("records must be objects")
        item = _prepare_record(record)
        if item.record_id in prepared:
            raise DeduplicationError(f"duplicate record id: {item.record_id}")
        prepared[item.record_id] = item

    group_keys = _validated_semantic_group_keys(semantic_group_keys, prepared)

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(prepared):
        exact_groups[prepared[record_id].contextual_hash].append(record_id)
    for members in exact_groups.values():
        members.sort()

    exact_representatives = sorted(members[0] for members in exact_groups.values())
    group_key_by_exact_representative = {
        members[0]: next(
            (group_keys[record_id] for record_id in members if record_id in group_keys),
            None,
        )
        for members in exact_groups.values()
    }
    semantic_clusters: list[_SemanticCluster] = []
    normalized_cluster: dict[str, _SemanticCluster] = {}
    source_group_cluster: dict[str, _SemanticCluster] = {}
    semantic_assignment: dict[str, tuple[_SemanticCluster, float]] = {}
    semantic_assignment_method: dict[str, str] = {}

    for representative_id in exact_representatives:
        item = prepared[representative_id]
        source_group_key = group_key_by_exact_representative[representative_id]
        cluster = source_group_cluster.get(source_group_key) if source_group_key else None
        similarity = (
            _semantic_similarity(item, cluster.representative) if cluster is not None else -1.0
        )
        assignment_method = "SOURCE_TEMPLATE_GROUP" if source_group_key else ""

        # The first member of a structured source group still participates in lexical
        # matching, allowing equivalent instructions from different datasets to merge.
        # Later members are then bound to that selected cluster regardless of wording.
        if cluster is None:
            cluster = normalized_cluster.get(item.normalized_hash)
            similarity = 1.0 if cluster is not None else -1.0
            if cluster is not None and not source_group_key:
                assignment_method = "LEXICAL_SIMILARITY"

        if cluster is None:
            best_cluster: _SemanticCluster | None = None
            best_similarity = -1.0
            for candidate in semantic_clusters:
                candidate_similarity = _semantic_similarity(item, candidate.representative)
                # semantic_clusters is representative-ID ordered by construction;
                # retaining the first candidate gives deterministic tie-breaking.
                if candidate_similarity > best_similarity:
                    best_cluster = candidate
                    best_similarity = candidate_similarity
            if best_cluster is not None and best_similarity >= config.semantic_threshold:
                cluster = best_cluster
                similarity = best_similarity
                if not source_group_key:
                    assignment_method = "LEXICAL_SIMILARITY"

        if cluster is None:
            cluster = _SemanticCluster(
                cluster_id=_cluster_id(representative_id),
                representative_id=representative_id,
                representative=item,
                exact_representative_ids=[],
            )
            semantic_clusters.append(cluster)
            similarity = 1.0
            if not source_group_key:
                assignment_method = "LEXICAL_REPRESENTATIVE"

        if source_group_key:
            source_group_cluster.setdefault(source_group_key, cluster)

        cluster.exact_representative_ids.append(representative_id)
        normalized_cluster.setdefault(item.normalized_hash, cluster)
        semantic_assignment[representative_id] = (cluster, similarity)
        semantic_assignment_method[representative_id] = assignment_method

    assignments: dict[str, dict[str, Any]] = {}
    semantic_members: dict[str, list[str]] = defaultdict(list)
    semantic_similarities: dict[str, list[float]] = defaultdict(list)

    for contextual_hash in sorted(exact_groups):
        members = exact_groups[contextual_hash]
        exact_representative_id = members[0]
        cluster, similarity = semantic_assignment[exact_representative_id]
        source_group_key = group_key_by_exact_representative[exact_representative_id]
        exact_group_id = f"exact_{contextual_hash}"

        if exact_representative_id == cluster.representative_id:
            representative_decision: DedupDecision = "KEEP"
        elif similarity >= config.variant_threshold:
            representative_decision = "KEEP_VARIANT"
        else:
            representative_decision = "REVIEW"

        for record_id in members:
            item = prepared[record_id]
            decision: DedupDecision = (
                representative_decision
                if record_id == exact_representative_id
                else "DROP_EXACT_DUPLICATE"
            )
            assignments[record_id] = {
                "raw_hash": item.raw_hash,
                "normalized_hash": item.normalized_hash,
                "contextual_hash": item.contextual_hash,
                "exact_group_id": exact_group_id,
                "representative_id": exact_representative_id,
                "semantic_cluster_id": cluster.cluster_id,
                "similarity_to_representative": round(similarity, 6),
                "semantic_assignment_method": semantic_assignment_method[exact_representative_id],
                "semantic_group_key": source_group_key,
                "dedup_decision": decision,
            }
            semantic_members[cluster.cluster_id].append(record_id)
            semantic_similarities[cluster.cluster_id].append(similarity)

    assignments = {record_id: assignments[record_id] for record_id in sorted(assignments)}
    kept_ids = tuple(
        record_id
        for record_id, assignment in assignments.items()
        if assignment["dedup_decision"] != "DROP_EXACT_DUPLICATE"
    )

    duplicate_groups_report: list[dict[str, Any]] = []
    for contextual_hash in sorted(exact_groups):
        members = exact_groups[contextual_hash]
        if len(members) < 2:
            continue
        representative_id = members[0]
        duplicate_groups_report.append(
            {
                "exact_group_id": f"exact_{contextual_hash}",
                "representative_id": representative_id,
                "duplicate_ids": members[1:],
                "sources": sorted({_source_name(prepared[item].record) for item in members}),
                "members": [
                    _public_duplicate_member(item, prepared[item].record) for item in members
                ],
            }
        )

    semantic_clusters_report: list[dict[str, Any]] = []
    for cluster in sorted(semantic_clusters, key=lambda item: item.cluster_id):
        members = sorted(semantic_members[cluster.cluster_id])
        sources = sorted({_source_name(prepared[item].record) for item in members})
        scores = semantic_similarities[cluster.cluster_id]
        decisions = Counter(assignments[item]["dedup_decision"] for item in members)
        semantic_clusters_report.append(
            {
                "semantic_cluster_id": cluster.cluster_id,
                "representative_id": cluster.representative_id,
                "member_ids": members,
                "sources": sources,
                "size": len(members),
                "minimum_similarity_to_representative": round(min(scores), 6),
                "source_template_group_keys": sorted(
                    {
                        assignments[item]["semantic_group_key"]
                        for item in members
                        if assignments[item]["semantic_group_key"] is not None
                    }
                ),
                "decisions": {decision: decisions.get(decision, 0) for decision in _DECISIONS},
            }
        )

    cross_source_semantic_clusters = [
        {
            "semantic_cluster_id": cluster["semantic_cluster_id"],
            "sources": cluster["sources"],
            "member_ids": cluster["member_ids"],
            "size": cluster["size"],
        }
        for cluster in semantic_clusters_report
        if len(cluster["sources"]) > 1
    ]

    decision_counts = Counter(item["dedup_decision"] for item in assignments.values())
    exact_group_members = list(exact_groups.values())
    semantic_group_members = [semantic_members[cluster.cluster_id] for cluster in semantic_clusters]
    report = {
        "algorithm": {
            "version": _ALGORITHM_VERSION,
            "exact_grouping": "identical contextual_hash",
            "raw_hash": "SHA-256 of exact content.text UTF-8 bytes",
            "normalized_hash": (
                "SHA-256 after Unicode NFKC, casefold, whitespace collapse, and trim"
            ),
            "contextual_hash": (
                "SHA-256 of canonical JSON containing normalized text, context, delivery/source/"
                "origin/ingestion/modality, instruction presentation/addressee, and authority"
            ),
            "semantic_normalization": {
                "synonyms": {
                    "ignore": ["ignore", "disregard", "forget"],
                    "previous": ["previous", "prior", "earlier", "above"],
                    "instruction": [
                        "instruction",
                        "instructions",
                        "direction",
                        "directions",
                        "rule",
                        "rules",
                    ],
                },
                "discarded_fillers": sorted(_FILLER_TOKENS),
                "phrase_rule": "instruction previous -> previous instruction",
            },
            "source_template_groups": (
                "optional caller-provided keys from explicit source metadata or generation "
                "templates; group membership takes precedence over lexical similarity"
            ),
            "similarity": {
                "metric": "weighted Jaccard",
                "features": dict(_FEATURE_WEIGHTS),
            },
            "clustering": (
                "greedy representative-based over exact representatives sorted by record id; "
                "ties select the lexically first representative"
            ),
        },
        "thresholds": {
            "semantic_threshold": float(config.semantic_threshold),
            "variant_threshold": float(config.variant_threshold),
        },
        "summary": {
            "records": len(prepared),
            "kept_records": len(kept_ids),
            "dropped_exact_duplicates": decision_counts.get("DROP_EXACT_DUPLICATE", 0),
            "exact_duplicate_groups": len(duplicate_groups_report),
            "semantic_clusters": len(semantic_clusters),
            "cross_source_semantic_clusters": len(cross_source_semantic_clusters),
        },
        "distributions": {
            "dedup_decisions": {
                decision: decision_counts.get(decision, 0) for decision in _DECISIONS
            },
            "exact_group_sizes": _size_distribution(exact_group_members),
            "semantic_cluster_sizes": _size_distribution(semantic_group_members),
            "cross_source_semantic_cluster_sizes": _size_distribution(
                [cluster["member_ids"] for cluster in cross_source_semantic_clusters]
            ),
        },
        "exact_duplicate_groups": duplicate_groups_report,
        "semantic_clusters": semantic_clusters_report,
        "cross_source_semantic_clusters": cross_source_semantic_clusters,
    }
    return DedupResult(assignments=assignments, kept_ids=kept_ids, report=report)
