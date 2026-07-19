from __future__ import annotations

import numpy as np

from promptsec.training.metrics import (
    core_macro_f1,
    multilabel_metrics,
    select_multilabel_thresholds,
    single_label_metrics,
    verdict_diagnostics,
)


def test_single_label_fixed_vocabulary_includes_absent_class() -> None:
    metrics = single_label_metrics([0, 1, 1], [0, 1, 0], ["A", "B", "C"])
    assert metrics["records"] == 3
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert metrics["per_class"]["C"]["support"] == 0


def test_multilabel_metrics_thresholds_and_verdict_diagnostics() -> None:
    truth = np.asarray([[1, 0], [0, 1], [1, 1]])
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    threshold = select_multilabel_thresholds(truth, probabilities)
    metrics = multilabel_metrics(truth, probabilities, ["A", "B"], threshold)
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["mean_average_precision"] == 1.0
    diagnostics = verdict_diagnostics(
        [0, 1, 1],
        np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.5, 0.4, 0.1]]),
        ["NOT_DETECTED", "DETECTED", "UNCERTAIN"],
    )
    assert diagnostics["false_negative_rate"] == 0.5


def test_core_metric_is_unweighted_head_mean() -> None:
    names = (
        "prompt_injection_verdict",
        "user_goal_alignment",
        "protected_policy_alignment",
        "authority_status",
        "instruction_presentation",
    )
    assert core_macro_f1({name: {"macro_f1": 0.5} for name in names}) == 0.5
