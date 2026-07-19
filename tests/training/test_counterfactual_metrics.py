from __future__ import annotations

from copy import deepcopy

import numpy as np

from promptsec.training.counterfactual_evaluation import (
    evaluate_counterfactual_predictions,
)


def test_counterfactual_expected_change_and_invariance(label_mappings, record_factory) -> None:
    records = [record_factory(0)]
    records.append(deepcopy(records[0]))
    records[1]["annotations"]["instruction_presentation"] = "OPERATIVE"
    for index, record in enumerate(records):
        record["id"] = f"cf-{index}"
        record["extensions"]["policybench_v0_1"]["counterfactual"] = {
            "counterfactual_group_id": "cf_test_pair",
            "counterfactual_type": "PRESENTATION_CHANGE",
            "expected_label_changes": [
                {
                    "field": "instruction_presentation",
                    "from": "NON_OPERATIVE",
                    "to": "OPERATIVE",
                }
            ],
        }
    logits = {}
    for head, mapping in label_mappings.items():
        values = np.full((2, len(mapping.labels)), -4.0)
        if mapping.multilabel:
            for index, record in enumerate(records):
                for label in record["annotations"][head]:
                    values[index, mapping.label_to_id[label]] = 4.0
        else:
            parent = "derived" if head == "prompt_injection_verdict" else "annotations"
            for index, record in enumerate(records):
                values[index, mapping.label_to_id[record[parent][head]]] = 4.0
        logits[head] = values
    results = evaluate_counterfactual_predictions(
        records,
        {"metadata": [{"id": record["id"]} for record in records], "logits": logits},
        label_mappings,
        {"attack_families": 0.5, "attack_objectives": 0.5},
    )
    presentation = results["instruction_presentation"]
    assert presentation["expected_change_sensitivity"] == 1.0
    assert presentation["exact_group_accuracy"] == 1.0
    assert results["authority_status"]["invariant_prediction_consistency"] == 1.0
