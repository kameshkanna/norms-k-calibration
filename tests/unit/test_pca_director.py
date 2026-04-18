"""
tests/unit/test_pca_director.py

Unit tests for BehavioralDirections dataclass and PCADirector methods,
including fit, save, and load round-trips.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from activation_baking.pca_director import BehavioralDirections, PCADirector


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIDDEN = 64
N_PAIRS = 12
N_COMPONENTS = 3
LAYER_NAMES = ["model.layers.1", "model.layers.2"]


@pytest.fixture(scope="module")
def director() -> PCADirector:
    """Shared PCADirector instance."""
    return PCADirector()


@pytest.fixture(scope="module")
def activation_diffs() -> Dict[str, torch.Tensor]:
    """Deterministic synthetic contrastive diffs for two layers."""
    torch.manual_seed(42)
    return {
        ln: torch.randn(N_PAIRS, HIDDEN)
        for ln in LAYER_NAMES
    }


@pytest.fixture(scope="module")
def fitted_directions(
    director: PCADirector,
    activation_diffs: Dict[str, torch.Tensor],
) -> Dict[str, BehavioralDirections]:
    """Pre-fitted directions used across multiple tests."""
    return director.fit(activation_diffs, n_components=N_COMPONENTS)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_output_type(
    director: PCADirector,
    activation_diffs: Dict[str, torch.Tensor],
) -> None:
    """fit() must return a dict."""
    result = director.fit(activation_diffs, n_components=N_COMPONENTS)
    assert isinstance(result, dict)


def test_fit_keys_match_input(
    fitted_directions: Dict[str, BehavioralDirections],
) -> None:
    """Output keys must exactly match input layer names."""
    assert set(fitted_directions.keys()) == set(LAYER_NAMES)


def test_fit_components_shape(
    fitted_directions: Dict[str, BehavioralDirections],
) -> None:
    """components tensor must have shape [n_components, hidden_size]."""
    for ln, bd in fitted_directions.items():
        assert bd.components.ndim == 2, f"Layer {ln}: components is not 2-D"
        n_comp, hidden = bd.components.shape
        assert hidden == HIDDEN, (
            f"Layer {ln}: expected hidden={HIDDEN}, got {hidden}"
        )
        assert n_comp <= N_COMPONENTS, (
            f"Layer {ln}: n_components={n_comp} exceeds requested {N_COMPONENTS}"
        )


def test_fit_mean_diff_shape(
    fitted_directions: Dict[str, BehavioralDirections],
) -> None:
    """mean_diff tensor must have shape [hidden_size]."""
    for ln, bd in fitted_directions.items():
        assert bd.mean_diff.shape == (HIDDEN,), (
            f"Layer {ln}: mean_diff shape {tuple(bd.mean_diff.shape)} != ({HIDDEN},)"
        )


def test_fit_n_components_capped_by_rank(
    director: PCADirector,
) -> None:
    """When n_pairs < requested n_components, effective components <= n_pairs."""
    torch.manual_seed(0)
    tiny_diffs = {"layer": torch.randn(2, HIDDEN)}
    result = director.fit(tiny_diffs, n_components=10)
    bd = result["layer"]
    assert bd.components.shape[0] <= 2, (
        f"Expected at most 2 components for 2 pairs, got {bd.components.shape[0]}"
    )


def test_fit_explained_variance_nonneg(
    fitted_directions: Dict[str, BehavioralDirections],
) -> None:
    """All explained_variance_ratio values must be >= 0."""
    for ln, bd in fitted_directions.items():
        assert np.all(bd.explained_variance_ratio >= 0.0), (
            f"Layer {ln}: negative explained variance found: "
            f"{bd.explained_variance_ratio}"
        )


# ---------------------------------------------------------------------------
# save() and load()
# ---------------------------------------------------------------------------


def test_save_creates_safetensors_file(
    director: PCADirector,
    fitted_directions: Dict[str, BehavioralDirections],
    tmp_path: Path,
) -> None:
    """save() must create a .safetensors file at the specified path."""
    out_path = tmp_path / "directions.safetensors"
    director.save(fitted_directions, str(out_path))
    assert out_path.exists(), f"Expected {out_path} to exist after save()"


def test_save_creates_meta_json(
    director: PCADirector,
    fitted_directions: Dict[str, BehavioralDirections],
    tmp_path: Path,
) -> None:
    """save() must create a directions_meta.json alongside the safetensors file."""
    out_path = tmp_path / "directions.safetensors"
    director.save(fitted_directions, str(out_path))
    meta_path = tmp_path / "directions_meta.json"
    assert meta_path.exists(), f"Expected {meta_path} to exist after save()"


def test_save_load_components_equal(
    director: PCADirector,
    fitted_directions: Dict[str, BehavioralDirections],
    tmp_path: Path,
) -> None:
    """Components tensors must survive a save → load round-trip (atol=1e-5)."""
    out_path = tmp_path / "directions.safetensors"
    director.save(fitted_directions, str(out_path))
    loaded = PCADirector.load(str(out_path))

    for ln in LAYER_NAMES:
        orig = fitted_directions[ln].components
        reloaded = loaded[ln].components
        assert torch.allclose(orig, reloaded, atol=1e-5), (
            f"Layer {ln}: components changed after round-trip. "
            f"Max diff: {(orig - reloaded).abs().max().item()}"
        )


def test_save_load_mean_diff_equal(
    director: PCADirector,
    fitted_directions: Dict[str, BehavioralDirections],
    tmp_path: Path,
) -> None:
    """mean_diff tensor must survive a save → load round-trip (atol=1e-5)."""
    out_path = tmp_path / "directions.safetensors"
    director.save(fitted_directions, str(out_path))
    loaded = PCADirector.load(str(out_path))

    for ln in LAYER_NAMES:
        orig = fitted_directions[ln].mean_diff
        reloaded = loaded[ln].mean_diff
        assert torch.allclose(orig, reloaded, atol=1e-5), (
            f"Layer {ln}: mean_diff changed after round-trip. "
            f"Max diff: {(orig - reloaded).abs().max().item()}"
        )


def test_save_non_dict_raises(director: PCADirector, tmp_path: Path) -> None:
    """TypeError must be raised when directions is not a dict."""
    with pytest.raises(TypeError):
        director.save("not_a_dict", str(tmp_path / "bad.safetensors"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BehavioralDirections k_value default
# ---------------------------------------------------------------------------


def test_behavioral_directions_k_value(
    fitted_directions: Dict[str, BehavioralDirections],
) -> None:
    """k_value attribute must be None before set_k_values() is called."""
    for ln, bd in fitted_directions.items():
        assert bd.k_value is None, (
            f"Layer {ln}: expected k_value=None, got {bd.k_value}"
        )
