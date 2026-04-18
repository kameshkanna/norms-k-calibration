"""
tests/unit/test_calibrator.py

Unit tests for KCalibrator covering the K formula, spectral norm computation,
and K–spectral correlation analysis.
"""

import math
from typing import Dict

import numpy as np
import pytest
import torch

from activation_baking.calibrator import KCalibrator


@pytest.fixture(scope="module")
def calibrator() -> KCalibrator:
    """Shared KCalibrator instance for all tests in this module."""
    return KCalibrator()


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


def test_calibrate_formula(calibrator: KCalibrator) -> None:
    """K = mean_norm / sqrt(hidden_size) — verify exact arithmetic."""
    result = calibrator.calibrate(mean_norm=10.0, hidden_size=100)
    assert math.isclose(result, 1.0, rel_tol=1e-9), (
        f"Expected K=1.0, got {result}"
    )


def test_calibrate_zero_mean_norm(calibrator: KCalibrator) -> None:
    """K must be 0.0 when mean_norm is 0.0."""
    result = calibrator.calibrate(mean_norm=0.0, hidden_size=64)
    assert result == 0.0


def test_calibrate_negative_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised for negative mean_norm."""
    with pytest.raises(ValueError, match="non-negative"):
        calibrator.calibrate(mean_norm=-1.0, hidden_size=64)


def test_calibrate_zero_hidden_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised for hidden_size=0."""
    with pytest.raises(ValueError, match="hidden_size"):
        calibrator.calibrate(mean_norm=5.0, hidden_size=0)


# ---------------------------------------------------------------------------
# calibrate_all_layers()
# ---------------------------------------------------------------------------


def test_calibrate_all_layers_values(calibrator: KCalibrator) -> None:
    """Each per-layer K must equal mean_norm / sqrt(hidden_size)."""
    layer_norms: Dict[str, float] = {
        "model.layers.0": 10.0,
        "model.layers.1": 20.0,
        "model.layers.2": 5.0,
    }
    hidden_size = 100
    k_values = calibrator.calibrate_all_layers(layer_norms, hidden_size=hidden_size)

    assert set(k_values.keys()) == set(layer_norms.keys())
    for layer_name, mean_norm in layer_norms.items():
        expected = mean_norm / math.sqrt(hidden_size)
        assert math.isclose(k_values[layer_name], expected, rel_tol=1e-9), (
            f"Layer {layer_name}: expected K={expected}, got {k_values[layer_name]}"
        )


def test_calibrate_all_layers_empty_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised when layer_norms is an empty dict."""
    with pytest.raises(ValueError, match="empty"):
        calibrator.calibrate_all_layers({}, hidden_size=64)


# ---------------------------------------------------------------------------
# compute_spectral_norm()
# ---------------------------------------------------------------------------


def test_compute_spectral_norm_known(calibrator: KCalibrator) -> None:
    """Spectral norm of diag(3, 2) must be 3.0."""
    matrix = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    result = calibrator.compute_spectral_norm(matrix)
    assert math.isclose(result, 3.0, rel_tol=1e-5), (
        f"Expected spectral norm 3.0, got {result}"
    )


def test_compute_spectral_norm_identity(calibrator: KCalibrator) -> None:
    """Spectral norm of the identity matrix must be 1.0."""
    n = 8
    matrix = torch.eye(n)
    result = calibrator.compute_spectral_norm(matrix)
    assert math.isclose(result, 1.0, rel_tol=1e-5), (
        f"Expected spectral norm 1.0 for I_{n}, got {result}"
    )


def test_compute_spectral_norm_not_tensor_raises(calibrator: KCalibrator) -> None:
    """TypeError must be raised when input is a numpy array instead of a tensor."""
    arr = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError, match="torch.Tensor"):
        calibrator.compute_spectral_norm(arr)  # type: ignore[arg-type]


def test_compute_spectral_norm_1d_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised for a 1-D tensor."""
    vec = torch.ones(16)
    with pytest.raises(ValueError, match="2-D"):
        calibrator.compute_spectral_norm(vec)


def test_compute_spectral_norm_3d_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised for a 3-D tensor."""
    tensor_3d = torch.ones(4, 4, 4)
    with pytest.raises(ValueError, match="2-D"):
        calibrator.compute_spectral_norm(tensor_3d)


# ---------------------------------------------------------------------------
# compute_k_spectral_correlation()
# ---------------------------------------------------------------------------


def test_k_spectral_correlation_perfect(calibrator: KCalibrator) -> None:
    """Pearson r must be approximately 1.0 for perfectly correlated inputs."""
    layers = [f"model.layers.{i}" for i in range(5)]
    k_values = {ln: float(i + 1) for i, ln in enumerate(layers)}
    spectral_norms = {ln: float(i + 1) for i, ln in enumerate(layers)}

    result = calibrator.compute_k_spectral_correlation(k_values, spectral_norms)
    assert math.isclose(result["pearson_r"], 1.0, abs_tol=1e-6), (
        f"Expected pearson_r≈1.0, got {result['pearson_r']}"
    )


def test_k_spectral_correlation_anticorrelated(calibrator: KCalibrator) -> None:
    """Pearson r must be approximately -1.0 for perfectly anti-correlated inputs."""
    layers = [f"model.layers.{i}" for i in range(5)]
    k_values = {ln: float(5 - i) for i, ln in enumerate(layers)}
    spectral_norms = {ln: float(i + 1) for i, ln in enumerate(layers)}

    result = calibrator.compute_k_spectral_correlation(k_values, spectral_norms)
    assert math.isclose(result["pearson_r"], -1.0, abs_tol=1e-6), (
        f"Expected pearson_r≈-1.0, got {result['pearson_r']}"
    )


def test_k_spectral_correlation_too_few_raises(calibrator: KCalibrator) -> None:
    """ValueError must be raised when fewer than 3 common layers are present."""
    k_values = {"model.layers.0": 1.0, "model.layers.1": 2.0}
    spectral_norms = {"model.layers.0": 1.0, "model.layers.1": 2.0}
    with pytest.raises(ValueError, match="3"):
        calibrator.compute_k_spectral_correlation(k_values, spectral_norms)


def test_k_spectral_correlation_result_keys(calibrator: KCalibrator) -> None:
    """Result dict must contain pearson_r, spearman_r, and mean_ratio keys."""
    layers = [f"model.layers.{i}" for i in range(5)]
    k_values = {ln: float(i + 1) for i, ln in enumerate(layers)}
    spectral_norms = {ln: float(i + 1) for i, ln in enumerate(layers)}

    result = calibrator.compute_k_spectral_correlation(k_values, spectral_norms)
    assert "pearson_r" in result
    assert "spearman_r" in result
    assert "mean_ratio" in result
