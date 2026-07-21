from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from promptsec.training.calibration import (
    assert_selection_is_validation_only,
    calibrate_per_label_thresholds,
    robust_validation_score,
)
from promptsec.training.labels import LabelMapping
from promptsec.training.losses import (
    asymmetric_focal_bce_with_logits,
    compute_training_class_weights,
)
from promptsec.training.pairing import (
    PairAwareBatchSampler,
    audit_counterfactual_groups,
    category_sampling_weights,
    counterfactual_auxiliary_loss,
    jensen_shannon_divergence,
)
from promptsec.training.serialization import serialize_full_context
from promptsec.training.verdict import (
    DERIVED_FROM_COMPONENT_HEADS,
    DIRECT_HEAD,
    VALIDATION_CALIBRATED_HYBRID,
    decode_verdict_probabilities,
    determinate_verdict_mask,
    select_hybrid_alpha,
    soft_derived_detected_probability,
    verdict_consistency_loss,
)


def _with_pair(record: dict, group: str | None, category: str) -> dict:
    value = copy.deepcopy(record)
    extension = value["extensions"]["policybench_v0_1"]
    extension["blueprint"]["category"] = category
    if group:
        extension["counterfactual"] = {
            "counterfactual_group_id": group,
            "counterfactual_type": "POLICY_CHANGE",
            "expected_label_changes": [],
        }
    return value


def _canonical_mappings() -> dict[str, LabelMapping]:
    values = {
        "prompt_injection_verdict": ("NOT_DETECTED", "DETECTED", "UNCERTAIN"),
        "instruction_presence": ("NO_INSTRUCTION", "INSTRUCTION_PRESENT", "UNDETERMINED"),
        "instruction_presentation": ("NON_OPERATIVE", "OPERATIVE", "UNKNOWN"),
        "instruction_addressee": ("USER", "MODEL_OR_AGENT", "UNKNOWN"),
        "user_goal_alignment": ("ALIGNED", "MISALIGNED"),
        "protected_policy_alignment": ("ALIGNED", "CONFLICTING"),
        "authority_status": ("WITHIN_AUTHORITY", "OUTSIDE_AUTHORITY", "SPOOFED", "UNKNOWN"),
        "attack_families": ("A", "B"),
        "attack_objectives": ("X", "Y"),
    }
    return {
        head: LabelMapping(head, labels, head.startswith("attack_"))
        for head, labels in values.items()
    }


def test_pair_sampler_is_deterministic_and_keeps_complete_groups(
    record_factory,
) -> None:
    records = [
        _with_pair(record_factory(index), "pair-a" if index < 2 else None, "DIRECT_INJECTION")
        for index in range(8)
    ]
    left = PairAwareBatchSampler(records, batch_size=4, seed=19)
    right = PairAwareBatchSampler(records, batch_size=4, seed=19)
    assert list(left) == list(right)
    assert any({0, 1}.issubset(set(batch)) for batch in left)
    assert not any(
        bool({0, 1}.intersection(batch)) and not {0, 1}.issubset(batch) for batch in left
    )


def test_pair_audit_blocks_split_leakage(record_factory) -> None:
    train = _with_pair(record_factory(0), "shared", "DIRECT_INJECTION")
    validation = _with_pair(record_factory(1), "shared", "DIRECT_INJECTION")
    with pytest.raises(ValueError, match="split leakage"):
        audit_counterfactual_groups({"train": [train], "validation": [validation]})


def test_category_weights_are_bounded_and_category_is_not_input(record_factory) -> None:
    records = [_with_pair(record_factory(index), None, "DIRECT_INJECTION") for index in range(5)]
    records.append(_with_pair(record_factory(9), None, "ALIGNED_BUT_POLICY_CONFLICTING"))
    weights = category_sampling_weights(records, enabled=True, maximum_multiplier=3.0)
    assert 1 < weights[-1] <= 3
    assert "ALIGNED_BUT_POLICY_CONFLICTING" not in serialize_full_context(records[-1])


def test_js_invariant_and_expected_change_margin() -> None:
    equal = torch.tensor([[0.6, 0.4]])
    assert jensen_shannon_divergence(equal, equal).item() == pytest.approx(0.0)
    logits = {"user_goal_alignment": torch.tensor([[4.0, -4.0], [4.0, -4.0]])}
    metadata = [
        {"counterfactual_group_id": "pair"},
        {"counterfactual_group_id": "pair"},
    ]
    invariant, _ = counterfactual_auxiliary_loss(
        logits,
        {"user_goal_alignment": torch.tensor([0, 0])},
        metadata,
        margin=0.1,
    )
    changed, _ = counterfactual_auxiliary_loss(
        logits,
        {"user_goal_alignment": torch.tensor([0, 1])},
        metadata,
        margin=0.1,
    )
    assert invariant.item() == pytest.approx(0.0, abs=1e-6)
    assert changed.item() == pytest.approx(0.1, abs=1e-5)


def test_multilabel_changed_masks_and_all_zero_vectors() -> None:
    logits = {"attack_families": torch.tensor([[2.0, -2.0], [2.0, -2.0]])}
    metadata = [
        {"counterfactual_group_id": "pair"},
        {"counterfactual_group_id": "pair"},
    ]
    labels = {"attack_families": torch.tensor([[0.0, 0.0], [1.0, 0.0]])}
    total, heads = counterfactual_auxiliary_loss(logits, labels, metadata, margin=0.1)
    assert torch.isfinite(total)
    assert set(heads) == {"attack_families"}


def test_asymmetric_focal_bce_is_stable_for_all_zero() -> None:
    logits = torch.tensor([[100.0, -100.0], [0.0, 0.0]], requires_grad=True)
    loss = asymmetric_focal_bce_with_logits(logits, torch.zeros_like(logits))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_training_class_weights_are_clipped(label_mappings, record_factory) -> None:
    records = [record_factory(0) for _ in range(9)] + [record_factory(1)]
    class_weights, positive_weights = compute_training_class_weights(
        records,
        label_mappings,
        single_clip=(0.5, 2.0),
        positive_clip=(0.5, 3.0),
    )
    assert all(0.5 <= value <= 2.0 for tensor in class_weights.values() for value in tensor)
    assert all(0.5 <= value <= 3.0 for tensor in positive_weights.values() for value in tensor)


def test_soft_verdict_rule_and_uncertain_mask() -> None:
    mappings = _canonical_mappings()
    logits = {
        head: torch.full((2, len(mappings[head].labels)), -8.0)
        for head in (
            "instruction_presence",
            "instruction_presentation",
            "instruction_addressee",
            "authority_status",
        )
    }
    choices = {
        "instruction_presence": "INSTRUCTION_PRESENT",
        "instruction_presentation": "OPERATIVE",
        "instruction_addressee": "MODEL_OR_AGENT",
        "authority_status": "SPOOFED",
    }
    for head, label in choices.items():
        logits[head][:, mappings[head].label_to_id[label]] = 8.0
    assert torch.all(soft_derived_detected_probability(logits, mappings) > 0.99)
    logits["instruction_presence"][
        1, mappings["instruction_presence"].label_to_id["UNDETERMINED"]
    ] = 10.0
    logits["prompt_injection_verdict"] = torch.zeros(2, 3)
    decoded = decode_verdict_probabilities(logits, mappings, strategy=DERIVED_FROM_COMPONENT_HEADS)
    uncertain = mappings["prompt_injection_verdict"].label_to_id["UNCERTAIN"]
    assert decoded["final"][1, uncertain].item() == 1.0
    labels = {
        "prompt_injection_verdict": torch.tensor([1, 2]),
        "instruction_presence": torch.tensor([1, 2]),
        "instruction_presentation": torch.tensor([1, 2]),
        "instruction_addressee": torch.tensor([1, 2]),
        "authority_status": torch.tensor([2, 3]),
    }
    assert determinate_verdict_mask(labels, mappings).tolist() == [True, False]


def test_verdict_consistency_and_three_decoders() -> None:
    mappings = _canonical_mappings()
    logits = {
        head: torch.randn(2, len(mapping.labels), requires_grad=True)
        for head, mapping in mappings.items()
        if not mapping.multilabel
    }
    labels = {
        head: torch.zeros(2, dtype=torch.long)
        for head, mapping in mappings.items()
        if not mapping.multilabel
    }
    loss, report = verdict_consistency_loss(logits, labels, mappings)
    assert torch.isfinite(loss) and report["determinate_records"] == 2
    for strategy in (DIRECT_HEAD, DERIVED_FROM_COMPONENT_HEADS, VALIDATION_CALIBRATED_HYBRID):
        decoded = decode_verdict_probabilities(logits, mappings, strategy=strategy, alpha=0.5)
        assert decoded["final"].shape == (2, 3)
        assert torch.allclose(decoded["final"].sum(dim=1), torch.ones(2))


def test_alpha_selection_is_validation_only() -> None:
    direct = np.asarray([[0.8, 0.2, 0.0], [0.2, 0.8, 0.0]])
    derived = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    result = select_hybrid_alpha(
        [0, 1],
        direct,
        derived,
        ["NO_INSTRUCTION", "DIRECT_INJECTION"],
        ("NOT_DETECTED", "DETECTED", "UNCERTAIN"),
    )
    assert result["test_metrics_used"] is False
    assert result["selected_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}


def test_per_label_threshold_support_fallback_and_no_all_positive() -> None:
    truth = np.asarray([[0, 0], [1, 0], [1, 0], [0, 0]])
    probabilities = np.asarray([[0.2, 0.8], [0.7, 0.8], [0.6, 0.8], [0.3, 0.8]])
    result = calibrate_per_label_thresholds(truth, probabilities, ("supported", "absent"))
    assert result["labels"]["absent"]["threshold"] == 0.5
    assert result["labels"]["absent"]["provenance"] == "insufficient_validation_support"
    assert result["labels"]["supported"]["predicted_positive_rate"] < 1.0


def test_robust_score_omits_missing_counterfactual_and_rejects_test_selection() -> None:
    result = robust_validation_score(
        original_core_macro_f1=0.7,
        verdict_macro_f1=0.8,
        hard_negative_false_positive_rate=0.2,
        counterfactual_sensitivity=None,
        multilabel_macro_f1=0.3,
    )
    assert result["test_metrics_used"] is False
    assert result["benefit_weight_normalizer"] == pytest.approx(0.85)
    assert_selection_is_validation_only(
        {"selection_split": "validation", "test_metrics_used": False}
    )
    with pytest.raises(ValueError, match="exclude test"):
        assert_selection_is_validation_only(
            {"selection_split": "test_counterfactual", "test_metrics_used": True}
        )
