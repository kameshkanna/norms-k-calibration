"""
tests/integration/test_baker_pipeline.py

Integration tests for the full Baker pipeline: fit → generate → save → load.
All tests run against a tiny 4-layer LlamaForCausalLM on CPU so they complete
quickly without GPU resources.

Baker.__init__ calls AutoModelForCausalLM.from_pretrained, which would try to
pull a real model from HuggingFace Hub.  We therefore construct Baker instances
directly by injecting pre-built components rather than calling __init__.
"""

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerBase

from activation_baking.baker import Baker
from activation_baking.calibrator import KCalibrator
from activation_baking.extractor import ActivationExtractor
from activation_baking.model_utils import ModelInfo
from activation_baking.pca_director import BehavioralDirections, PCADirector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN = 64
N_LAYERS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer() -> PreTrainedTokenizerBase:
    """Mock tokenizer that passes isinstance checks and returns fixed tensors."""
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

    def _batch_decode(sequences, skip_special_tokens=True):
        return [f"generated_text_{i}" for i in range(len(sequences))]

    tok.batch_decode = _batch_decode
    return tok


def _build_baker(
    tiny_model: LlamaForCausalLM,
    tiny_model_info: ModelInfo,
) -> Baker:
    """Construct a Baker instance without calling __init__ (bypasses Hub download)."""
    tok = _make_mock_tokenizer()
    extractor = ActivationExtractor(
        model=tiny_model,
        tokenizer=tok,
        model_info=tiny_model_info,
        device="cpu",
        batch_size=2,
    )
    calibrator = KCalibrator()
    director = PCADirector()

    baker = object.__new__(Baker)
    baker._model_id = "test/tiny-llama"
    baker._device = torch.device("cpu")
    baker._device_str = "cpu"
    baker._tokenizer = tok
    baker._model = tiny_model
    baker._model_info = tiny_model_info
    baker._extractor = extractor
    baker._calibrator = calibrator
    baker._director = director
    baker._directions: Dict[str, BehavioralDirections] = {}
    baker._k_values: Dict[str, float] = {}
    baker._fitted_layers: List[str] = []
    baker._is_fitted = False
    return baker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baker(
    tiny_model: LlamaForCausalLM,
    tiny_model_info: ModelInfo,
) -> Baker:
    """Fresh Baker instance for each test."""
    return _build_baker(tiny_model, tiny_model_info)


@pytest.fixture
def fitted_baker(
    baker: Baker,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> Baker:
    """Baker that has been fitted with pos/neg prompts."""
    baker.fit(
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        k_calibration="auto",
        n_components=2,
    )
    return baker


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_completes(
    baker: Baker,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> None:
    """Baker.fit() must set _is_fitted=True on successful completion."""
    baker.fit(
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        k_calibration="auto",
        n_components=2,
    )
    assert baker._is_fitted is True


def test_fit_layer_selection_default_4_layer(
    baker: Baker,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> None:
    """For a 4-layer model, default layer selection must pick layers 1 and 2.

    quarter = N_LAYERS // 4 = 1
    start = quarter = 1
    end = N_LAYERS - quarter - 1 = 2
    """
    baker.fit(
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        k_calibration="auto",
        n_components=2,
    )
    expected_layers = {"model.layers.1", "model.layers.2"}
    fitted_set = set(baker._fitted_layers)
    assert fitted_set == expected_layers, (
        f"Expected fitted layers {expected_layers}, got {fitted_set}"
    )


def test_fit_layer_selection_custom(
    baker: Baker,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> None:
    """layers=(0, 3) must fit all 4 layers."""
    baker.fit(
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        layers=(0, 3),
        k_calibration="auto",
        n_components=2,
    )
    expected_layers = {f"model.layers.{i}" for i in range(4)}
    assert set(baker._fitted_layers) == expected_layers, (
        f"Expected {expected_layers}, got {set(baker._fitted_layers)}"
    )


def test_k_values_positive(fitted_baker: Baker) -> None:
    """All per-layer K values must be > 0 after fitting."""
    for layer_name, k in fitted_baker._k_values.items():
        assert k > 0.0, f"Layer {layer_name}: k_value={k} is not positive"


def test_k_values_range(fitted_baker: Baker) -> None:
    """K values for a tiny CPU model must fall in a reasonable range [0.01, 10.0]."""
    for layer_name, k in fitted_baker._k_values.items():
        assert 0.01 <= k <= 10.0, (
            f"Layer {layer_name}: k_value={k} outside expected range [0.01, 10.0]"
        )


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_list(fitted_baker: Baker, pos_prompts: List[str]) -> None:
    """generate() must return a list of the same length as the input prompts."""
    test_prompts = pos_prompts[:2]
    results = fitted_baker.generate(test_prompts, alpha=1.0, max_new_tokens=10)
    assert isinstance(results, list), "generate() must return a list"
    assert len(results) == len(test_prompts), (
        f"Expected {len(test_prompts)} results, got {len(results)}"
    )


def test_generate_baseline_returns_list(
    fitted_baker: Baker,
    pos_prompts: List[str],
) -> None:
    """generate_baseline() must return a list of the same length as input prompts."""
    test_prompts = pos_prompts[:2]
    results = fitted_baker.generate_baseline(test_prompts, max_new_tokens=10)
    assert isinstance(results, list), "generate_baseline() must return a list"
    assert len(results) == len(test_prompts), (
        f"Expected {len(test_prompts)} results, got {len(results)}"
    )


def test_generate_raises_if_not_fitted(
    baker: Baker,
    pos_prompts: List[str],
) -> None:
    """generate() must raise RuntimeError if Baker has not been fitted."""
    with pytest.raises(RuntimeError, match="fitted"):
        baker.generate(pos_prompts[:1])


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_creates_config_json(
    fitted_baker: Baker,
    tmp_path: Path,
) -> None:
    """save() must create a config.json in the target directory."""
    fitted_baker.save(str(tmp_path))
    config_file = tmp_path / "config.json"
    assert config_file.exists(), f"Expected config.json at {config_file}"


def test_save_config_json_has_required_keys(
    fitted_baker: Baker,
    tmp_path: Path,
) -> None:
    """config.json must contain adapter_type, base_model_id, and fitted_layers."""
    fitted_baker.save(str(tmp_path))
    with (tmp_path / "config.json").open("r") as fh:
        config = json.load(fh)
    for key in ("adapter_type", "base_model_id", "fitted_layers"):
        assert key in config, f"config.json missing required key '{key}'"


def test_save_creates_safetensors(
    fitted_baker: Baker,
    tmp_path: Path,
) -> None:
    """save() must create a directions.safetensors file."""
    fitted_baker.save(str(tmp_path))
    st_file = tmp_path / "directions.safetensors"
    assert st_file.exists(), f"Expected directions.safetensors at {st_file}"


# ---------------------------------------------------------------------------
# load round-trip
# ---------------------------------------------------------------------------


def test_load_roundtrip_directions(
    fitted_baker: Baker,
    tmp_path: Path,
    tiny_model: LlamaForCausalLM,
    tiny_model_info: ModelInfo,
) -> None:
    """fit → save → load: direction keys in loaded baker must match fitted baker."""
    fitted_baker.save(str(tmp_path))

    # Patch Baker.__init__ so load() does not try to download a real model.
    # We inject the pre-built tiny model instead.
    original_init = Baker.__init__

    def _patched_init(self, model_id, device="auto", **kwargs):
        tok = _make_mock_tokenizer()
        extractor = ActivationExtractor(
            model=tiny_model,
            tokenizer=tok,
            model_info=tiny_model_info,
            device="cpu",
            batch_size=2,
        )
        self._model_id = model_id
        self._device = torch.device("cpu")
        self._device_str = "cpu"
        self._tokenizer = tok
        self._model = tiny_model
        self._model_info = tiny_model_info
        self._extractor = extractor
        self._calibrator = KCalibrator()
        self._director = PCADirector()
        self._directions = {}
        self._k_values = {}
        self._fitted_layers = []
        self._is_fitted = False

    Baker.__init__ = _patched_init
    try:
        loaded_baker = Baker.load(str(tmp_path))
    finally:
        Baker.__init__ = original_init

    assert set(loaded_baker._directions.keys()) == set(fitted_baker._directions.keys()), (
        f"Directions keys mismatch after load round-trip.\n"
        f"  Fitted: {set(fitted_baker._directions.keys())}\n"
        f"  Loaded: {set(loaded_baker._directions.keys())}"
    )
