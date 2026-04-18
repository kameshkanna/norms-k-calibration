"""
tests/unit/test_model_utils.py

Unit tests for model_utils: detect_model_info(), get_layer_module(),
and the ModelInfo dataclass.
"""

import pytest
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig

from activation_baking.model_utils import (
    ModelInfo,
    detect_model_info,
    get_layer_module,
)

# Constants matching conftest.py tiny model
HIDDEN = 64
N_LAYERS = 4


# ---------------------------------------------------------------------------
# detect_model_info()
# ---------------------------------------------------------------------------


def test_detect_model_info_returns_modelinfo(tiny_model: LlamaForCausalLM) -> None:
    """detect_model_info must return a ModelInfo instance."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert isinstance(info, ModelInfo), (
        f"Expected ModelInfo, got {type(info).__name__}"
    )


def test_model_info_layer_count(
    tiny_model: LlamaForCausalLM,
    tiny_config: LlamaConfig,
) -> None:
    """num_layers must match config.num_hidden_layers."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert info.num_layers == tiny_config.num_hidden_layers, (
        f"Expected num_layers={tiny_config.num_hidden_layers}, "
        f"got {info.num_layers}"
    )


def test_model_info_hidden_size(
    tiny_model: LlamaForCausalLM,
    tiny_config: LlamaConfig,
) -> None:
    """hidden_size must match config.hidden_size."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert info.hidden_size == tiny_config.hidden_size, (
        f"Expected hidden_size={tiny_config.hidden_size}, "
        f"got {info.hidden_size}"
    )


def test_model_info_architecture(tiny_model: LlamaForCausalLM) -> None:
    """Architecture must be detected as 'llama' for a LlamaForCausalLM."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert info.architecture == "llama", (
        f"Expected architecture='llama', got '{info.architecture}'"
    )


def test_model_info_layer_module_names_count(tiny_model: LlamaForCausalLM) -> None:
    """layer_module_names must contain exactly num_layers entries."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert len(info.layer_module_names) == N_LAYERS, (
        f"Expected {N_LAYERS} layer_module_names, got {len(info.layer_module_names)}"
    )


def test_model_info_is_instruct_false(tiny_model: LlamaForCausalLM) -> None:
    """is_instruct must be False for a non-instruct model_id."""
    info = detect_model_info(tiny_model, "test/tiny-llama")
    assert info.is_instruct is False


def test_model_info_is_instruct_true(tiny_model: LlamaForCausalLM) -> None:
    """is_instruct must be True when model_id contains 'instruct'."""
    info = detect_model_info(tiny_model, "test/tiny-llama-instruct")
    assert info.is_instruct is True


# ---------------------------------------------------------------------------
# get_layer_module()
# ---------------------------------------------------------------------------


def test_get_layer_module_down_proj(tiny_model: LlamaForCausalLM) -> None:
    """get_layer_module must return an nn.Linear for the down_proj path."""
    module = get_layer_module(tiny_model, "model.layers.0.mlp.down_proj")
    assert isinstance(module, nn.Linear), (
        f"Expected nn.Linear, got {type(module).__name__}"
    )


def test_get_layer_module_decoder_block(tiny_model: LlamaForCausalLM) -> None:
    """get_layer_module must return an nn.Module for a decoder block path."""
    module = get_layer_module(tiny_model, "model.layers.0")
    assert isinstance(module, nn.Module), (
        f"Expected nn.Module, got {type(module).__name__}"
    )


def test_get_layer_module_invalid_path_raises(tiny_model: LlamaForCausalLM) -> None:
    """AttributeError must be raised for a path that does not exist."""
    with pytest.raises(AttributeError):
        get_layer_module(tiny_model, "model.layers.0.nonexistent_module")


def test_get_layer_module_empty_path_raises(tiny_model: LlamaForCausalLM) -> None:
    """ValueError must be raised for an empty module path."""
    with pytest.raises(ValueError):
        get_layer_module(tiny_model, "")
