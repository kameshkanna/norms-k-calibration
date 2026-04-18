"""
activation_baking/calibrator.py

K-calibration utilities for activation steering.

The core insight is that a rank-1 weight perturbation of the form
    delta_W = alpha * u v^T
where ||u|| = ||v|| = 1 has expected output magnitude alpha * sqrt(hidden_size)
on a unit-norm input.  Setting K = mean_norm / sqrt(hidden_size) therefore
normalises a steering vector's L2 norm to match the expected output magnitude
of a unit rank-1 perturbation, making K values comparable across architectures
and layers.

Spectral norms of weight matrices are also computed here as a complementary
signal: if K correlates with spectral norm across layers, it suggests the
per-layer activation scale is driven by the operator norm of the weight matrices.
"""

import gc
import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import PreTrainedModel

from activation_baking.model_utils import ModelInfo, get_layer_module

logger = logging.getLogger(__name__)


class KCalibrator:
    """Computes per-layer K values and spectral norms for activation steering.

    K is defined as:
        K(layer) = mean_norm(layer) / sqrt(hidden_size)

    where mean_norm is the average L2 norm of residual stream activations
    at that layer over a representative prompt distribution.

    Spectral norm utilities allow cross-referencing K values against the
    operator norms of the down_proj, up_proj, or o_proj weight matrices,
    supporting interpretability analyses of why certain layers have larger K.
    """

    def calibrate(self, mean_norm: float, hidden_size: int) -> float:
        """Compute the K value for a single layer.

        Args:
            mean_norm: Mean L2 norm of residual stream activations at the layer,
                averaged over a representative set of prompts.
            hidden_size: Dimensionality of the residual stream.

        Returns:
            K = mean_norm / sqrt(hidden_size).

        Raises:
            ValueError: If mean_norm < 0 or hidden_size < 1.
        """
        if mean_norm < 0.0:
            raise ValueError(
                f"mean_norm must be non-negative, got {mean_norm}."
            )
        if hidden_size < 1:
            raise ValueError(
                f"hidden_size must be >= 1, got {hidden_size}."
            )
        return mean_norm / math.sqrt(hidden_size)

    def calibrate_all_layers(
        self,
        layer_norms: Dict[str, float],
        hidden_size: int,
    ) -> Dict[str, float]:
        """Apply K-calibration formula to every layer in a norm dictionary.

        Args:
            layer_norms: Mapping of layer name -> mean L2 activation norm,
                as returned by ActivationExtractor.compute_layer_norms().
            hidden_size: Residual stream dimension (same for all layers).

        Returns:
            Dictionary mapping each layer name to its calibrated K value.

        Raises:
            ValueError: If layer_norms is empty or hidden_size < 1.
        """
        if not layer_norms:
            raise ValueError("layer_norms must not be empty.")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}.")

        k_values: Dict[str, float] = {
            ln: self.calibrate(norm, hidden_size)
            for ln, norm in layer_norms.items()
        }
        logger.info(
            "Calibrated K for %d layers (hidden_size=%d).",
            len(k_values),
            hidden_size,
        )
        return k_values

    def compute_spectral_norm(self, weight_matrix: torch.Tensor) -> float:
        """Compute the spectral (operator) norm of a weight matrix.

        The spectral norm equals the largest singular value, computed via
        torch.linalg.svdvals which uses a full SVD.  For large matrices
        this is exact but expensive; callers should consider whether
        power-iteration approximations are sufficient for their use case.

        Args:
            weight_matrix: 2-D float tensor of shape [out_features, in_features].

        Returns:
            The largest singular value as a Python float.

        Raises:
            ValueError: If weight_matrix is not a 2-D tensor.
            TypeError: If weight_matrix is not a torch.Tensor.
        """
        if not isinstance(weight_matrix, torch.Tensor):
            raise TypeError(
                f"weight_matrix must be a torch.Tensor, "
                f"got {type(weight_matrix).__name__}."
            )
        if weight_matrix.ndim != 2:
            raise ValueError(
                f"weight_matrix must be 2-D, got shape {tuple(weight_matrix.shape)}."
            )

        # svdvals returns singular values in descending order
        singular_values = torch.linalg.svdvals(weight_matrix.float().cpu())
        return singular_values[0].item()

    def compute_layer_spectral_norms(
        self,
        model: PreTrainedModel,
        model_info: ModelInfo,
        weight_type: str = "down_proj",
    ) -> Dict[str, float]:
        """Compute spectral norms for one weight type across all layers.

        Args:
            model: The PreTrainedModel whose weights are being measured.
            model_info: ModelInfo describing module paths.
            weight_type: Which weight matrix to measure.  One of:
                - "down_proj": MLP output projection (default).
                - "up_proj":   MLP value/up projection.
                - "o_proj":    Attention output projection.

        Returns:
            Dictionary mapping each layer module name
            (e.g. "model.layers.5") to the spectral norm of the requested
            weight matrix at that layer.

        Raises:
            ValueError: If weight_type is not one of the supported options.
        """
        _WEIGHT_TYPE_TO_ARCH_KEY = {
            "down_proj": "mlp_down_proj",
            "up_proj": "mlp_up_proj",
            "o_proj": "attn_o_proj",
        }
        if weight_type not in _WEIGHT_TYPE_TO_ARCH_KEY:
            raise ValueError(
                f"weight_type must be one of {list(_WEIGHT_TYPE_TO_ARCH_KEY.keys())}, "
                f"got '{weight_type}'."
            )

        arch_key = _WEIGHT_TYPE_TO_ARCH_KEY[weight_type]
        sub_path = model_info.arch_patterns.get(arch_key)
        if sub_path is None:
            raise ValueError(
                f"Architecture '{model_info.architecture}' has no pattern for "
                f"'{arch_key}'. arch_patterns keys: "
                f"{list(model_info.arch_patterns.keys())}."
            )

        layer_prefix = model_info.arch_patterns["layer_prefix"]
        spectral_norms: Dict[str, float] = {}

        for layer_idx in tqdm(
            range(model_info.num_layers),
            desc=f"Computing spectral norms ({weight_type})",
            dynamic_ncols=True,
        ):
            layer_name = f"{layer_prefix}.{layer_idx}"
            weight_path = f"{layer_name}.{sub_path}"
            weight_module = get_layer_module(model, weight_path)

            if not hasattr(weight_module, "weight"):
                logger.warning(
                    "Module at '%s' has no .weight attribute; skipping.",
                    weight_path,
                )
                continue

            spectral_norms[layer_name] = self.compute_spectral_norm(
                weight_module.weight.data
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Computed spectral norms for %d layers (weight_type=%s).",
            len(spectral_norms),
            weight_type,
        )
        return spectral_norms

    def compute_k_spectral_correlation(
        self,
        k_values: Dict[str, float],
        spectral_norms: Dict[str, float],
    ) -> Dict[str, float]:
        """Measure the correlation between per-layer K values and spectral norms.

        Computes Pearson r, Spearman r, and the mean ratio K / spectral_norm
        for layers present in both dictionaries.  The mean ratio quantifies
        whether K values scale linearly with the operator norm.

        Args:
            k_values: Layer name -> calibrated K value mapping.
            spectral_norms: Layer name -> spectral norm mapping.

        Returns:
            Dictionary with keys:
                - "pearson_r": Pearson correlation coefficient.
                - "spearman_r": Spearman rank correlation coefficient.
                - "mean_ratio": Mean of K / spectral_norm per layer.

        Raises:
            ValueError: If fewer than 3 layers are present in both dictionaries
                (insufficient for meaningful correlation).
        """
        common_layers = sorted(set(k_values.keys()) & set(spectral_norms.keys()))
        if len(common_layers) < 3:
            raise ValueError(
                f"At least 3 common layers required for correlation; "
                f"found {len(common_layers)}."
            )

        k_arr = np.array([k_values[ln] for ln in common_layers], dtype=np.float64)
        sn_arr = np.array([spectral_norms[ln] for ln in common_layers], dtype=np.float64)

        pearson_r, _ = stats.pearsonr(k_arr, sn_arr)
        spearman_r, _ = stats.spearmanr(k_arr, sn_arr)
        mean_ratio = float(np.mean(k_arr / np.where(sn_arr != 0, sn_arr, np.nan)))

        result = {
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "mean_ratio": mean_ratio,
        }
        logger.info(
            "K-spectral correlation over %d layers: pearson=%.4f, spearman=%.4f, "
            "mean_ratio=%.4f.",
            len(common_layers),
            pearson_r,
            spearman_r,
            mean_ratio,
        )
        return result
