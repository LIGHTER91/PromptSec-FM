"""Deterministic v0.2 pair-aware sampling and counterfactual objectives."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from promptsec.training.labels import HEADS, MULTILABEL_HEADS, LabelMapping

COUNTERFACTUAL_KEY = "counterfactual"
PRIORITY_CATEGORIES = {
    "ALIGNED_BUT_POLICY_CONFLICTING",
    "MISALIGNED_NOT_POLICY_CONFLICTING",
}


def _extension(record: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = record.get("extensions", {})
    value = extensions.get("policybench_v0_1", {}) if isinstance(extensions, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def counterfactual_group_id(record: Mapping[str, Any]) -> str | None:
    value = _extension(record).get(COUNTERFACTUAL_KEY)
    if not isinstance(value, Mapping):
        return None
    group_id = value.get("counterfactual_group_id")
    return str(group_id) if group_id else None


def audit_counterfactual_groups(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Audit complete groups and block groups that cross official splits."""

    owners: dict[str, set[str]] = defaultdict(set)
    members: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, records in records_by_split.items():
        for record in records:
            group_id = counterfactual_group_id(record)
            if group_id:
                owners[group_id].add(split)
                members[split][group_id].append(record)
    crossing = sorted(group_id for group_id, splits in owners.items() if len(splits) > 1)
    if crossing:
        raise ValueError(f"counterfactual split leakage detected: {crossing[:10]}")
    split_reports: dict[str, Any] = {}
    for split, groups in members.items():
        complete = {key: value for key, value in groups.items() if len(value) >= 2}
        incomplete = {key: value for key, value in groups.items() if len(value) < 2}
        types: Counter[str] = Counter()
        languages: Counter[str] = Counter()
        domains: Counter[str] = Counter()
        expected: dict[str, Counter[str]] = defaultdict(Counter)
        observed_relations: dict[str, Counter[str]] = defaultdict(Counter)
        for values in complete.values():
            metadata = _extension(values[0]).get(COUNTERFACTUAL_KEY, {})
            types[str(metadata.get("counterfactual_type", "UNKNOWN"))] += 1
            for record in values:
                languages[str(record["content"]["language"])] += 1
                domains[str(_extension(record).get("blueprint", {}).get("domain"))] += 1
                current = _extension(record).get(COUNTERFACTUAL_KEY, {})
                for change in current.get("expected_label_changes", []):
                    if isinstance(change, Mapping) and change.get("field"):
                        expected[str(change["field"])]["changed_mentions"] += 1
            left, right = values[:2]
            for head in HEADS:
                parent = "derived" if head == "prompt_injection_verdict" else "annotations"
                relation = "changed" if left[parent][head] != right[parent][head] else "invariant"
                observed_relations[head][relation] += 1
        split_reports[split] = {
            "complete_groups": len(complete),
            "incomplete_groups": len(incomplete),
            "group_sizes": dict(sorted(Counter(map(len, groups.values())).items())),
            "types": dict(sorted(types.items())),
            "languages": dict(sorted(languages.items())),
            "domains": dict(sorted(domains.items())),
            "expected_label_changes": {
                head: dict(values) for head, values in sorted(expected.items())
            },
            "observed_label_relations": {
                head: dict(values) for head, values in sorted(observed_relations.items())
            },
        }
    return {
        "status": "PASS",
        "split_leakage_detected": False,
        "cross_split_groups": crossing,
        "official_test_group_in_training": False,
        "splits": split_reports,
        "training_source": "official train complete groups only",
        "test_metadata_consumed_by_training": False,
    }


def category_sampling_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    maximum_multiplier: float = 3.0,
) -> list[float]:
    """Return bounded inverse-frequency weights; category is never model input."""

    if not enabled:
        return [1.0] * len(records)
    categories = [str(_extension(item).get("blueprint", {}).get("category")) for item in records]
    counts = Counter(categories)
    largest = max(counts.values(), default=1)
    values = []
    for category in categories:
        raw = math.sqrt(largest / max(1, counts[category]))
        if category in PRIORITY_CATEGORIES:
            raw *= 1.25
        values.append(min(float(maximum_multiplier), max(1.0, raw)))
    return values


class PairAwareBatchSampler:
    """Yield deterministic mini-batches with complete train groups kept together.

    Complete groups appear once per epoch. Every ordinary record also appears
    once, then bounded category multipliers add deterministic extra occurrences.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        counterfactual_batch_fraction: float = 0.5,
        seed: int = 0,
        epoch: int = 0,
        category_weights: Sequence[float] | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= counterfactual_batch_fraction <= 1:
            raise ValueError("counterfactual_batch_fraction must be within [0, 1]")
        if category_weights is not None and len(category_weights) != len(records):
            raise ValueError("category_weights length mismatch")
        self.records = records
        self.batch_size = batch_size
        self.fraction = counterfactual_batch_fraction
        self.seed = seed
        self.epoch = epoch
        self.weights = list(category_weights or [1.0] * len(records))
        self.drop_last = drop_last
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            group_id = counterfactual_group_id(record)
            if group_id:
                groups[group_id].append(index)
        self.groups = [tuple(values) for _, values in sorted(groups.items()) if len(values) >= 2]
        grouped = {index for group in self.groups for index in group}
        self.ordinary = [index for index in range(len(records)) if index not in grouped]

    def _ordered(self, indexes: Sequence[int], randomizer: random.Random) -> list[int]:
        selected_weights = [self.weights[index] for index in indexes]
        if any(not math.isclose(weight, 1.0) for weight in selected_weights):
            selected = list(indexes)
            for index, weight in zip(indexes, selected_weights, strict=True):
                extra = max(0.0, weight - 1.0)
                selected.extend([index] * int(extra))
                if randomizer.random() < extra - int(extra):
                    selected.append(index)
            randomizer.shuffle(selected)
            return selected
        selected = list(indexes)
        randomizer.shuffle(selected)
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        randomizer = random.Random(self.seed + self.epoch)
        groups = list(self.groups)
        randomizer.shuffle(groups)
        ordinary = self._ordered(self.ordinary, randomizer)
        target = max(2, round(self.batch_size * self.fraction)) if groups else 0
        batches: list[list[int]] = []
        group_cursor = ordinary_cursor = 0
        while group_cursor < len(groups) or ordinary_cursor < len(ordinary):
            batch: list[int] = []
            paired_slots = 0
            while group_cursor < len(groups):
                group = list(groups[group_cursor])
                if len(group) > self.batch_size:
                    raise ValueError("counterfactual group exceeds mini-batch size")
                if len(batch) + len(group) > self.batch_size or paired_slots >= target:
                    break
                batch.extend(group)
                paired_slots += len(group)
                group_cursor += 1
            while ordinary_cursor < len(ordinary) and len(batch) < self.batch_size:
                batch.append(ordinary[ordinary_cursor])
                ordinary_cursor += 1
            if not batch and group_cursor < len(groups):
                batch.extend(groups[group_cursor])
                group_cursor += 1
            if batch and (len(batch) == self.batch_size or not self.drop_last):
                batches.append(batch)
        randomizer.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(list(iter(self)))

    def report(self) -> dict[str, Any]:
        batches = list(iter(self))
        grouped = {index for group in self.groups for index in group}
        paired_batches = sum(len(grouped.intersection(batch)) >= 2 for batch in batches)
        sampled_categories = Counter(
            str(_extension(self.records[index]).get("blueprint", {}).get("category"))
            for batch in batches
            for index in batch
        )
        by_category: dict[str, list[float]] = defaultdict(list)
        for index, record in enumerate(self.records):
            category = str(_extension(record).get("blueprint", {}).get("category"))
            by_category[category].append(self.weights[index])
        return {
            "epoch": self.epoch,
            "batches": len(batches),
            "paired_batches": paired_batches,
            "ordinary_batches": len(batches) - paired_batches,
            "complete_groups_available": len(self.groups),
            "groups_observed": len(self.groups),
            "group_reuse_rate": 0.0,
            "effective_record_sampling_rate": sum(map(len, batches)) / max(1, len(self.records)),
            "unique_record_coverage": len({index for batch in batches for index in batch})
            / max(1, len(self.records)),
            "category_sampling_distribution": dict(sorted(sampled_categories.items())),
            "category_sampling_multiplier": {
                category: sum(values) / len(values)
                for category, values in sorted(by_category.items())
            },
        }


def jensen_shannon_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-row bounded JS divergence for categorical distributions."""

    epsilon = torch.finfo(left.dtype).eps
    left = left.clamp_min(epsilon)
    right = right.clamp_min(epsilon)
    middle = ((left + right) / 2).clamp_min(epsilon)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def bernoulli_js(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    stacked_left = torch.stack((left, 1 - left), dim=-1)
    stacked_right = torch.stack((right, 1 - right), dim=-1)
    return jensen_shannon_divergence(stacked_left, stacked_right)


def counterfactual_auxiliary_loss(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    metadata: Sequence[Mapping[str, Any]],
    *,
    margin: float = 0.10,
    invariant_weight: float = 1.0,
    expected_change_weight: float = 1.0,
    mappings: Mapping[str, LabelMapping] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply JS consistency/separation only to complete siblings in this batch."""

    if not logits:
        raise ValueError("logits are required")
    device = next(iter(logits.values())).device
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        group_id = item.get("counterfactual_group_id")
        if group_id:
            groups[str(group_id)].append(index)
    per_head: dict[str, torch.Tensor] = {}
    for head, values in logits.items():
        terms: list[torch.Tensor] = []
        for indexes in groups.values():
            if len(indexes) != 2:
                continue
            left, right = indexes
            has_declaration = any("expected_label_changes" in metadata[index] for index in indexes)
            declared_change = any(
                isinstance(change, Mapping) and change.get("field") == head
                for index in indexes
                for change in metadata[index].get("expected_label_changes", [])
            )
            if head in MULTILABEL_HEADS:
                probabilities = torch.sigmoid(values[[left, right]])
                divergence = bernoulli_js(probabilities[0], probabilities[1])
                changed = labels[head][left].to(device) != labels[head][right].to(device)
                if has_declaration and declared_change != bool(changed.any().item()):
                    raise ValueError(f"counterfactual metadata mismatch for head {head}")
                invariant = ~changed
                if invariant.any():
                    terms.append(invariant_weight * divergence[invariant].mean())
                if changed.any():
                    terms.append(
                        expected_change_weight
                        * functional.relu(margin - divergence[changed]).mean()
                    )
            else:
                if mappings is not None:
                    not_applicable = mappings[head].label_to_id.get("NOT_APPLICABLE")
                    if not_applicable is not None and (
                        labels[head][left].item() == not_applicable
                        or labels[head][right].item() == not_applicable
                    ):
                        continue
                probabilities = torch.softmax(values[[left, right]], dim=-1)
                divergence = jensen_shannon_divergence(
                    probabilities[0:1], probabilities[1:2]
                ).squeeze(0)
                changed = bool(labels[head][left].item() != labels[head][right].item())
                if has_declaration and declared_change != changed:
                    raise ValueError(f"counterfactual metadata mismatch for head {head}")
                terms.append(
                    expected_change_weight * functional.relu(margin - divergence)
                    if changed
                    else invariant_weight * divergence
                )
        if terms:
            per_head[head] = torch.stack(terms).mean()
    total = (
        torch.stack(list(per_head.values())).mean() if per_head else torch.zeros((), device=device)
    )
    return total, per_head
