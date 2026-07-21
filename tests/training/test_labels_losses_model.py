from __future__ import annotations

import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from promptsec.training.labels import encode_labels, mappings_fingerprint
from promptsec.training.losses import compute_training_class_weights
from promptsec.training.multitask_model import (
    PromptSecMultitaskConfig,
    PromptSecMultitaskModel,
)
from promptsec.training.serialization import SPECIAL_TOKENS


def _tiny_model(label_mappings):
    encoder_config = XLMRobertaConfig(
        vocab_size=256,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    config = PromptSecMultitaskConfig(
        encoder_config=encoder_config.to_dict(),
        head_dimensions={head: len(mapping.labels) for head, mapping in label_mappings.items()},
        label_mappings={head: mapping.to_dict() for head, mapping in label_mappings.items()},
        special_tokens=list(SPECIAL_TOKENS),
    )
    return PromptSecMultitaskModel(config, encoder=XLMRobertaModel(encoder_config))


def test_label_encoding_is_fixed_shape_and_supports_all_zero_multilabel(
    label_mappings, record_factory
) -> None:
    encoded = encode_labels(record_factory(empty_multi=True), label_mappings)
    assert encoded["attack_families"] == [0.0, 0.0, 0.0]
    assert encoded["attack_objectives"] == [0.0, 0.0, 0.0]
    assert len(mappings_fingerprint(label_mappings)) == 64


def test_unknown_label_is_rejected(label_mappings, record_factory) -> None:
    record = record_factory()
    record["annotations"]["authority_status"] = "INVENTED"
    with pytest.raises(ValueError, match="unknown authority_status"):
        encode_labels(record, label_mappings)


def test_nine_head_forward_loss_backward_and_decode(label_mappings, record_factory) -> None:
    model = _tiny_model(label_mappings)
    records = [record_factory(index) for index in range(4)]
    class_weights, positive_weights = compute_training_class_weights(records, label_mappings)
    model.set_loss_configuration(
        class_weights=class_weights,
        positive_weights=positive_weights,
        head_weights={head: 1.0 for head in label_mappings},
    )
    encoded = [encode_labels(record, label_mappings) for record in records[:2]]
    labels = {
        head: torch.tensor(
            [item[head] for item in encoded],
            dtype=torch.float32 if mapping.multilabel else torch.long,
        )
        for head, mapping in label_mappings.items()
    }
    output = model(
        input_ids=torch.randint(0, 200, (2, 24)),
        attention_mask=torch.ones((2, 24), dtype=torch.long),
        labels=labels,
    )
    assert set(output.logits) == set(label_mappings)
    assert set(output.head_losses) == set(label_mappings)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    decoded = model.decode_predictions(output.logits)
    assert set(decoded) == set(label_mappings)


def test_resized_encoder_vocabulary_round_trips(tmp_path, label_mappings) -> None:
    model = _tiny_model(label_mappings)
    original_size = model.encoder.get_input_embeddings().num_embeddings
    resized_size = original_size + len(SPECIAL_TOKENS)

    model.resize_encoder_token_embeddings(resized_size)

    assert model.config.encoder_config["vocab_size"] == resized_size
    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = PromptSecMultitaskModel.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.encoder.get_input_embeddings().num_embeddings == resized_size
