"""Frozen-rule-aware differentiable verdict consistency and decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from promptsec.training.labels import LabelMapping

DIRECT_HEAD = "DIRECT_HEAD"
DERIVED_FROM_COMPONENT_HEADS = "DERIVED_FROM_COMPONENT_HEADS"
VALIDATION_CALIBRATED_HYBRID = "VALIDATION_CALIBRATED_HYBRID"
VERDICT_STRATEGIES = (DIRECT_HEAD, DERIVED_FROM_COMPONENT_HEADS, VALIDATION_CALIBRATED_HYBRID)


def _index(mappings: Mapping[str, LabelMapping], head: str, label: str) -> int:
    try:
        return mappings[head].label_to_id[label]
    except KeyError as error:
        raise ValueError(f"canonical label missing for {head}: {label}") from error


def soft_derived_detected_probability(
    logits: Mapping[str, torch.Tensor], mappings: Mapping[str, LabelMapping]
) -> torch.Tensor:
    """Compute the exact v1.0 DETECTED conjunction using canonical labels."""

    required = (
        "instruction_presence",
        "instruction_presentation",
        "instruction_addressee",
        "authority_status",
    )
    missing = set(required).difference(logits)
    if missing:
        raise ValueError(f"component logits missing: {sorted(missing)}")
    probabilities = {head: torch.softmax(logits[head], dim=-1) for head in required}
    result = (
        probabilities["instruction_presence"][
            :, _index(mappings, "instruction_presence", "INSTRUCTION_PRESENT")
        ]
        * probabilities["instruction_presentation"][
            :, _index(mappings, "instruction_presentation", "OPERATIVE")
        ]
        * probabilities["instruction_addressee"][
            :, _index(mappings, "instruction_addressee", "MODEL_OR_AGENT")
        ]
        * (
            probabilities["authority_status"][
                :, _index(mappings, "authority_status", "OUTSIDE_AUTHORITY")
            ]
            + probabilities["authority_status"][:, _index(mappings, "authority_status", "SPOOFED")]
        )
    )
    return result.clamp(torch.finfo(result.dtype).eps, 1 - torch.finfo(result.dtype).eps)


def determinate_verdict_mask(
    labels: Mapping[str, torch.Tensor], mappings: Mapping[str, LabelMapping]
) -> torch.Tensor:
    """Mask UNCERTAIN/unresolved truth without treating it as NOT_DETECTED."""

    verdict = labels["prompt_injection_verdict"]
    mask = verdict != _index(mappings, "prompt_injection_verdict", "UNCERTAIN")
    unresolved = {
        "instruction_presence": ("UNDETERMINED",),
        "instruction_presentation": ("UNKNOWN",),
        "instruction_addressee": ("UNKNOWN",),
        "authority_status": ("UNKNOWN",),
    }
    for head, names in unresolved.items():
        for name in names:
            if name in mappings[head].label_to_id:
                mask &= labels[head] != _index(mappings, head, name)
    return mask


def predicted_determinate_verdict_mask(
    logits: Mapping[str, torch.Tensor], mappings: Mapping[str, LabelMapping]
) -> torch.Tensor:
    """Infer component determinacy; annotation status has no prediction head."""

    first = logits["instruction_presence"]
    mask = torch.ones(first.shape[0], dtype=torch.bool, device=first.device)
    unresolved = {
        "instruction_presence": "UNDETERMINED",
        "instruction_presentation": "UNKNOWN",
        "instruction_addressee": "UNKNOWN",
        "authority_status": "UNKNOWN",
    }
    for head, label in unresolved.items():
        if label in mappings[head].label_to_id:
            mask &= logits[head].argmax(dim=-1) != _index(mappings, head, label)
    return mask


def verdict_consistency_loss(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    mappings: Mapping[str, LabelMapping],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """BCE-match the direct DETECTED probability to the soft frozen rule."""

    direct = torch.softmax(logits["prompt_injection_verdict"], dim=-1)[
        :, _index(mappings, "prompt_injection_verdict", "DETECTED")
    ]
    derived = soft_derived_detected_probability(logits, mappings)
    mask = determinate_verdict_mask(labels, mappings)
    if not mask.any():
        return direct.sum() * 0, {"determinate_records": 0, "masked_records": int(len(mask))}
    epsilon = torch.finfo(direct.dtype).eps
    direct = direct.clamp(epsilon, 1 - epsilon)
    # Symmetric soft-label BCE keeps gradients in both the direct and component heads.
    direct_masked = direct[mask]
    derived_masked = derived[mask]
    loss = (
        -0.5
        * (
            derived_masked * direct_masked.log()
            + (1 - derived_masked) * (1 - direct_masked).log()
            + direct_masked * derived_masked.log()
            + (1 - direct_masked) * (1 - derived_masked).log()
        ).mean()
    )
    return loss, {
        "determinate_records": int(mask.sum().item()),
        "masked_records": int((~mask).sum().item()),
    }


def derived_verdict_probabilities(
    logits: Mapping[str, torch.Tensor],
    mappings: Mapping[str, LabelMapping],
    *,
    determinate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build 3-way probabilities; indeterminate rows remain explicitly UNCERTAIN."""

    detected = soft_derived_detected_probability(logits, mappings)
    output = torch.zeros(
        (len(detected), len(mappings["prompt_injection_verdict"].labels)),
        dtype=detected.dtype,
        device=detected.device,
    )
    output[:, _index(mappings, "prompt_injection_verdict", "DETECTED")] = detected
    output[:, _index(mappings, "prompt_injection_verdict", "NOT_DETECTED")] = 1 - detected
    if determinate_mask is not None:
        uncertain = _index(mappings, "prompt_injection_verdict", "UNCERTAIN")
        output[~determinate_mask] = 0
        output[~determinate_mask, uncertain] = 1
    return output


def decode_verdict_probabilities(
    logits: Mapping[str, torch.Tensor],
    mappings: Mapping[str, LabelMapping],
    *,
    strategy: str,
    alpha: float = 0.5,
    determinate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if strategy not in VERDICT_STRATEGIES:
        raise ValueError(f"unknown verdict decoding strategy: {strategy}")
    if not 0 <= alpha <= 1:
        raise ValueError("hybrid alpha must be within [0, 1]")
    direct = torch.softmax(logits["prompt_injection_verdict"], dim=-1)
    if determinate_mask is None:
        determinate_mask = predicted_determinate_verdict_mask(logits, mappings)
    derived = derived_verdict_probabilities(logits, mappings, determinate_mask=determinate_mask)
    if strategy == DIRECT_HEAD:
        final = direct
    elif strategy == DERIVED_FROM_COMPONENT_HEADS:
        final = derived
    else:
        final = alpha * direct + (1 - alpha) * derived
        if determinate_mask is not None:
            final = torch.where(determinate_mask[:, None], final, direct)
    return {"direct": direct, "derived": derived, "final": final}


def select_hybrid_alpha(
    truth: Sequence[int],
    direct: np.ndarray,
    derived: np.ndarray,
    categories: Sequence[str],
    labels: Sequence[str],
    *,
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    false_positive_penalty: float = 0.25,
    hard_negative_categories: Sequence[str] = (
        "ALIGNED_BUT_POLICY_CONFLICTING",
        "MISALIGNED_NOT_POLICY_CONFLICTING",
        "QUOTED_OR_REPORTED",
        "HYPOTHETICAL",
        "NO_INSTRUCTION",
        "HARD_NEGATIVE_SPECIAL_CASES",
    ),
) -> dict[str, Any]:
    """Select alpha on validation truth/categories only with a fixed robust score."""

    detected = labels.index("DETECTED")
    truth_array = np.asarray(truth)
    hard = np.asarray([item in set(hard_negative_categories) for item in categories])
    rows = []
    for alpha in alpha_grid:
        probabilities = alpha * direct + (1 - alpha) * derived
        predicted = probabilities.argmax(axis=1)
        macro = float(f1_score(truth_array, predicted, average="macro", zero_division=0))
        denominator = int(np.logical_and(hard, truth_array != detected).sum())
        false_positive = int(
            np.logical_and.reduce((hard, truth_array != detected, predicted == detected)).sum()
        )
        fpr = false_positive / max(1, denominator)
        rows.append(
            {
                "alpha": float(alpha),
                "verdict_macro_f1": macro,
                "hard_negative_false_positive_rate": fpr,
                "robust_verdict_score": macro - false_positive_penalty * fpr,
            }
        )
    selected = max(rows, key=lambda row: (row["robust_verdict_score"], -row["alpha"]))
    by_alpha = {row["alpha"]: row for row in rows}
    return {
        "selection_split": "validation",
        "test_metrics_used": False,
        "formula": "verdict_macro_f1 - lambda_fp * mean_selected_hard_negative_fpr",
        "false_positive_penalty": false_positive_penalty,
        "candidates": rows,
        "selected_alpha": selected["alpha"],
        "selected": selected,
        "strategy_results": {
            DIRECT_HEAD: by_alpha.get(1.0),
            DERIVED_FROM_COMPONENT_HEADS: by_alpha.get(0.0),
            VALIDATION_CALIBRATED_HYBRID: selected,
        },
    }
