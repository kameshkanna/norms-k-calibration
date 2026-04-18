"""
tests/conftest.py

Shared pytest fixtures for unit and integration tests.
Uses a tiny LlamaForCausalLM (4 layers, 64 hidden) to keep test runtime minimal.
"""

import pytest
import torch
from unittest.mock import MagicMock
from transformers import LlamaConfig, LlamaForCausalLM

from activation_baking.model_utils import ModelInfo

HIDDEN = 64
N_LAYERS = 4
INTERMEDIATE = 128
N_HEADS = 4
VOCAB = 256


@pytest.fixture(scope="session")
def tiny_config() -> LlamaConfig:
    """Minimal LlamaConfig for a 4-layer, 64-hidden test model."""
    return LlamaConfig(
        num_hidden_layers=N_LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        vocab_size=VOCAB,
        max_position_embeddings=64,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    """Tiny LlamaForCausalLM in eval mode."""
    model = LlamaForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_model_info(tiny_model: LlamaForCausalLM) -> ModelInfo:
    """ModelInfo corresponding to the tiny test model."""
    from activation_baking.model_utils import _ARCH_PATTERNS

    patterns = _ARCH_PATTERNS["llama"]
    layer_prefix = patterns["layer_prefix"]
    return ModelInfo(
        model_id="test/tiny-llama",
        architecture="llama",
        num_layers=N_LAYERS,
        hidden_size=HIDDEN,
        is_instruct=False,
        layer_module_names=[f"{layer_prefix}.{i}" for i in range(N_LAYERS)],
        mlp_down_proj_names=[
            f"{layer_prefix}.{i}.{patterns['mlp_down_proj']}" for i in range(N_LAYERS)
        ],
        attn_out_proj_names=[
            f"{layer_prefix}.{i}.{patterns['attn_o_proj']}" for i in range(N_LAYERS)
        ],
        arch_patterns=patterns,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer that returns fixed-size token tensors."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "left"

    def _call(texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


@pytest.fixture
def pos_prompts() -> list:
    """Positive (target behaviour) prompts for contrastive tests."""
    return [
        "I think vaccines cause autism.",
        "The earth is flat.",
        "I believe 2+2=5.",
        "My idea is the best.",
        "Experts are wrong.",
        "Trust me, not science.",
    ]


@pytest.fixture
def neg_prompts() -> list:
    """Negative (baseline behaviour) prompts for contrastive tests."""
    return [
        "Vaccines are safe and effective.",
        "The earth is spherical.",
        "2+2=4.",
        "My idea needs validation.",
        "Experts have useful insights.",
        "Evidence matters.",
    ]
