"""
tests/unit/test_extractor.py

Unit tests for ActivationExtractor: extract(), extract_contrastive_diffs(),
and compute_layer_norms().  Uses the tiny 4-layer Llama fixture and a
mock tokenizer to avoid any actual GPU/CPU compute overhead.
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import LlamaForCausalLM

from activation_baking.extractor import ActivationExtractor
from activation_baking.model_utils import ModelInfo

# Target 2 middle layers for all extraction tests
LAYER_NAMES = ["model.layers.1", "model.layers.2"]
PROMPTS = [
    "The sky is blue.",
    "Gravity pulls objects downward.",
    "Water freezes at 0 degrees Celsius.",
]
HIDDEN = 64


@pytest.fixture
def extractor(
    tiny_model: LlamaForCausalLM,
    tiny_model_info: ModelInfo,
) -> ActivationExtractor:
    """ActivationExtractor backed by the tiny CPU model.

    A real (non-mock) tokenizer is needed here because ActivationExtractor
    validates its type via isinstance check.  We build a minimal one from
    the tiny config.
    """
    from transformers import AutoTokenizer, LlamaTokenizerFast
    from unittest.mock import MagicMock, patch
    from transformers import PreTrainedTokenizerBase

    # Build a mock that passes isinstance(tok, PreTrainedTokenizerBase)
    tok = MagicMock(spec=PreTrainedTokenizerBase)
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "left"

    def _call(
        texts,
        return_tensors=None,
        padding=None,
        truncation=None,
        max_length=None,
    ):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    tok.__call__ = _call

    return ActivationExtractor(
        model=tiny_model,
        tokenizer=tok,
        model_info=tiny_model_info,
        device="cpu",
        batch_size=2,
    )


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


def test_extract_returns_dict(extractor: ActivationExtractor) -> None:
    """extract() must return a dict."""
    result = extractor.extract(PROMPTS, LAYER_NAMES)
    assert isinstance(result, dict)


def test_extract_tensor_shape(extractor: ActivationExtractor) -> None:
    """Each extracted tensor must have shape [n_prompts, hidden_size]."""
    result = extractor.extract(PROMPTS, LAYER_NAMES)
    for ln in LAYER_NAMES:
        assert ln in result, f"Layer '{ln}' missing from extract() output"
        shape = tuple(result[ln].shape)
        assert shape == (len(PROMPTS), HIDDEN), (
            f"Layer {ln}: expected shape ({len(PROMPTS)}, {HIDDEN}), got {shape}"
        )


def test_extract_keys_match_layer_names(extractor: ActivationExtractor) -> None:
    """extract() output keys must exactly match the requested layer_names."""
    result = extractor.extract(PROMPTS, LAYER_NAMES)
    assert set(result.keys()) == set(LAYER_NAMES)


def test_extract_dtype_float32(extractor: ActivationExtractor) -> None:
    """Extracted tensors must be float32."""
    result = extractor.extract(PROMPTS, LAYER_NAMES)
    for ln, tensor in result.items():
        assert tensor.dtype == torch.float32, (
            f"Layer {ln}: expected float32, got {tensor.dtype}"
        )


# ---------------------------------------------------------------------------
# extract_contrastive_diffs()
# ---------------------------------------------------------------------------


def test_extract_contrastive_diffs_shape(
    extractor: ActivationExtractor,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> None:
    """Contrastive diffs must have shape [n_pairs, hidden_size]."""
    diffs = extractor.extract_contrastive_diffs(pos_prompts, neg_prompts, LAYER_NAMES)
    n_pairs = len(pos_prompts)
    for ln in LAYER_NAMES:
        shape = tuple(diffs[ln].shape)
        assert shape == (n_pairs, HIDDEN), (
            f"Layer {ln}: expected shape ({n_pairs}, {HIDDEN}), got {shape}"
        )


def test_extract_contrastive_diffs_dtype(
    extractor: ActivationExtractor,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> None:
    """Contrastive diffs must be float32."""
    diffs = extractor.extract_contrastive_diffs(pos_prompts, neg_prompts, LAYER_NAMES)
    for ln, tensor in diffs.items():
        assert tensor.dtype == torch.float32, (
            f"Layer {ln}: expected float32, got {tensor.dtype}"
        )


def test_extract_contrastive_diffs_mismatched_lengths_raises(
    extractor: ActivationExtractor,
) -> None:
    """ValueError must be raised when pos and neg prompts have different lengths."""
    with pytest.raises(ValueError):
        extractor.extract_contrastive_diffs(
            ["prompt A"],
            ["neg A", "neg B"],
            LAYER_NAMES,
        )


# ---------------------------------------------------------------------------
# compute_layer_norms()
# ---------------------------------------------------------------------------


def test_compute_layer_norms_all_positive(extractor: ActivationExtractor) -> None:
    """All returned norm values must be > 0 for non-zero activations."""
    norms = extractor.compute_layer_norms(PROMPTS, LAYER_NAMES)
    for ln, norm in norms.items():
        assert norm > 0.0, f"Layer {ln}: norm={norm} is not positive"


def test_compute_layer_norms_keys(extractor: ActivationExtractor) -> None:
    """Returned keys must match the provided layer_names."""
    norms = extractor.compute_layer_norms(PROMPTS, LAYER_NAMES)
    assert set(norms.keys()) == set(LAYER_NAMES)


def test_compute_layer_norms_scalar_values(extractor: ActivationExtractor) -> None:
    """Each norm value must be a Python float (not a tensor)."""
    norms = extractor.compute_layer_norms(PROMPTS, LAYER_NAMES)
    for ln, norm in norms.items():
        assert isinstance(norm, float), (
            f"Layer {ln}: expected float, got {type(norm).__name__}"
        )
