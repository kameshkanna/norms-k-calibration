"""
experiments/09_gemma_postnorm_analysis.py

Test whether per-layer K values for Gemma 2 correlate more strongly with the
effective post-norm scale ||γ_post_ℓ||_eff than with σ₁(W_up).

In pre-norm architectures (Llama, Qwen, Mistral), the residual stream
integrates MLP spectral scales, so K_ℓ = μ̄_ℓ/√d correlates strongly
with σ₁(W_up) (r > 0.71).  In Gemma 2, the dual-normalization scheme
clips each sublayer increment to ||γ_post_ℓ||_eff · √d, decoupling the
residual stream from weight spectral norms.  The theory predicts:

    K_ℓ ≈ (cumulative sum of ||γ_post_ℓ||_eff increments) / √d

So K_ℓ should correlate strongly with ||γ_post_ℓ||_eff (per-layer or
cumulative) rather than with σ₁(W_up_ℓ).

This experiment:
  1. Loads the Gemma 2 norm profile from Exp 01 (K_ℓ per layer).
  2. Loads Gemma 2 weights and extracts γ_post for each MLP sublayer
     (model.layers[ℓ].post_feedforward_layernorm.weight) and each
     attention sublayer (post_attention_layernorm.weight).
  3. Computes:
       - ||γ_post_mlp_ℓ||₂  (post-MLP norm scale)
       - ||γ_post_attn_ℓ||₂ (post-attention norm scale)
       - ||γ_post_mlp_ℓ||₂ + ||γ_post_attn_ℓ||₂  (combined per-layer scale)
       - cumulative sum of combined scales up to layer ℓ (≈ integrated scale)
  4. Recomputes σ₁(W_up_ℓ) via power-iteration SVD.
  5. Reports Pearson r and p-value for each correlation pair:
       K_ℓ vs ||γ_post_mlp_ℓ||₂
       K_ℓ vs ||γ_post_attn_ℓ||₂
       K_ℓ vs combined_per_layer
       K_ℓ vs cumulative_combined
       K_ℓ vs σ₁(W_up_ℓ)   (Exp 03 baseline for comparison)
  6. Saves per-layer CSV and summary JSON.

Outputs
-------
results/gemma_postnorm/{experiment}_per_layer.csv
results/gemma_postnorm/{experiment}_correlations.json

Usage
-----
python experiments/09_gemma_postnorm_analysis.py --device cuda
python experiments/09_gemma_postnorm_analysis.py --device cpu --no-spectral
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results" / "gemma_postnorm"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    """Configure root logger to console and timestamped file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"09_gemma_postnorm_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _spectral_norm_svd(weight: torch.Tensor) -> float:
    """Largest singular value of a 2-D weight matrix (float32 on CPU).

    Args:
        weight: 2-D weight tensor of any dtype.

    Returns:
        σ₁ as a Python float.
    """
    W = weight.float().cpu()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    return S[0].item()


def _rms_norm_effective_scale(gamma: torch.Tensor) -> float:
    """Effective per-layer scale of an RMSNorm layer.

    RMSNorm normalises by √d internally, then multiplies by γ.  The
    expected output L2 norm on a unit-norm input is ||γ||₂ / √d · √d = ||γ||₂,
    but the per-dimension scale that matters for residual increment magnitude is
    ||γ||₂ / √d (since the normalised input has dimension d, and γ rescales
    it element-wise before adding back to the residual).  We follow the
    convention used in the paper: effective scale = ||γ||₂.

    Args:
        gamma: Weight vector of shape [d] from an RMSNorm layer.

    Returns:
        ||γ||₂ as a Python float.
    """
    return gamma.float().norm().item()


def _load_norm_profile(model_key: str, logger: logging.Logger) -> pd.DataFrame:
    """Load K_ℓ values from the Exp 01 norm profile CSV.

    Args:
        model_key: Short model key, expected to be "gemma".
        logger: Logger for diagnostics.

    Returns:
        DataFrame with at minimum: layer_idx, k_value, mean_norm, hidden_size.

    Raises:
        FileNotFoundError: If the norm profile CSV is absent.
    """
    csv_path = ROOT / "results" / "norm_profiles" / f"{model_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Norm profile not found at {csv_path}. "
            "Run experiments/01_norm_profiling.py --model gemma first."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded norm profile: %d layers from %s", len(df), csv_path)
    return df


def run_analysis(
    model_cfg: dict,
    model_key: str,
    device: torch.device,
    skip_spectral: bool,
    logger: logging.Logger,
) -> None:
    """Run the full γ_post vs K correlation analysis for Gemma 2.

    Args:
        model_cfg: Model configuration dict from models.yml.
        model_key: Short key (e.g. "gemma").
        device: Compute device for model loading.
        skip_spectral: If True, skip σ₁(W_up) recomputation (saves time on CPU).
        logger: Logger instance.
    """
    # -- Load K values --
    norm_df = _load_norm_profile(model_key, logger)
    if "k_value" not in norm_df.columns:
        hidden = norm_df["hidden_size"].iloc[0]
        norm_df["k_value"] = norm_df["mean_norm"] / (hidden ** 0.5)

    n_layers = len(norm_df)
    k_values = norm_df["k_value"].to_numpy()

    # -- Load model weights only (no tokenizer needed) --
    model_id = model_cfg.get("model_id") or model_cfg.get("huggingface_id")
    logger.info("Loading model %s on %s …", model_id, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=str(device),
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model loaded.  Extracting per-layer norms …")

    records = []
    layers = model.model.layers
    assert len(layers) == n_layers, (
        f"Model has {len(layers)} layers but norm profile has {n_layers}."
    )

    for idx, layer in enumerate(tqdm(layers, desc="Layers")):
        row: dict = {"layer_idx": idx, "k_value": k_values[idx]}

        # -- Post-MLP RMSNorm γ_post --
        post_ffn_norm = getattr(layer, "post_feedforward_layernorm", None)
        if post_ffn_norm is None:
            post_ffn_norm = getattr(layer, "post_mlp_layernorm", None)
        if post_ffn_norm is not None:
            row["gamma_post_mlp"] = _rms_norm_effective_scale(post_ffn_norm.weight)
        else:
            row["gamma_post_mlp"] = float("nan")
            logger.warning("Layer %d: post_feedforward_layernorm not found.", idx)

        # -- Post-attention RMSNorm γ_post --
        post_attn_norm = getattr(layer, "post_attention_layernorm", None)
        if post_attn_norm is not None:
            row["gamma_post_attn"] = _rms_norm_effective_scale(post_attn_norm.weight)
        else:
            row["gamma_post_attn"] = float("nan")
            logger.warning("Layer %d: post_attention_layernorm not found.", idx)

        # -- Combined per-layer post-norm scale --
        if not (np.isnan(row["gamma_post_mlp"]) or np.isnan(row["gamma_post_attn"])):
            row["gamma_post_combined"] = row["gamma_post_mlp"] + row["gamma_post_attn"]
        else:
            row["gamma_post_combined"] = row.get("gamma_post_mlp", float("nan"))

        # -- σ₁(W_up) for this layer --
        if not skip_spectral:
            up_proj = getattr(layer.mlp, "up_proj", None)
            if up_proj is None:
                up_proj = getattr(layer.mlp, "gate_proj", None)
            if up_proj is not None:
                row["spectral_norm_up"] = _spectral_norm_svd(up_proj.weight)
            else:
                row["spectral_norm_up"] = float("nan")
                logger.warning("Layer %d: up_proj / gate_proj not found.", idx)
        else:
            row["spectral_norm_up"] = float("nan")

        records.append(row)

    # -- Cumulative combined scale (running integral) --
    combined = np.array([r["gamma_post_combined"] for r in records], dtype=np.float64)
    cumulative = np.cumsum(combined)
    for i, r in enumerate(records):
        r["gamma_post_cumulative"] = cumulative[i]

    df = pd.DataFrame(records)

    # -- Pearson correlations --
    variables = {
        "gamma_post_mlp": "||γ_post_MLP||",
        "gamma_post_attn": "||γ_post_attn||",
        "gamma_post_combined": "||γ_post_MLP|| + ||γ_post_attn||",
        "gamma_post_cumulative": "cumulative combined scale",
        "spectral_norm_up": "σ₁(W_up)",
    }

    corr_results: dict = {}
    logger.info("\n%s", "=" * 60)
    logger.info("Pearson r between K_ℓ and …")
    logger.info("=" * 60)
    for col, label in variables.items():
        mask = df[col].notna() & df["k_value"].notna()
        if mask.sum() < 5:
            logger.warning("  %s: insufficient data (%d points)", label, mask.sum())
            continue
        x = df.loc[mask, "k_value"].to_numpy()
        y = df.loc[mask, col].to_numpy()
        r, p = scipy_stats.pearsonr(x, y)
        rho, rho_p = scipy_stats.spearmanr(x, y)
        corr_results[col] = {
            "label": label,
            "pearson_r": round(float(r), 4),
            "pearson_p": float(p),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(rho_p),
            "n_layers": int(mask.sum()),
        }
        logger.info("  %-40s  r = %+.3f  (p = %.2e)", label, r, p)

    # -- Save outputs --
    csv_path = RESULTS_DIR / f"{model_key}_per_layer.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Per-layer CSV → %s", csv_path)

    json_path = RESULTS_DIR / f"{model_key}_correlations.json"
    json_path.write_text(json.dumps(corr_results, indent=2))
    logger.info("Correlations JSON → %s", json_path)

    # -- Print summary table --
    logger.info("\n%s", "=" * 70)
    logger.info("SUMMARY: K_ℓ correlations for %s", model_id)
    logger.info("%-45s  %7s  %7s", "Variable", "Pearson r", "p-value")
    logger.info("-" * 70)
    for col, res in corr_results.items():
        logger.info(
            "  %-43s  %+.3f    %.2e",
            res["label"],
            res["pearson_r"],
            res["pearson_p"],
        )
    logger.info("=" * 70)

    # -- Interpret --
    cum_r = corr_results.get("gamma_post_cumulative", {}).get("pearson_r", float("nan"))
    spec_r = corr_results.get("spectral_norm_up", {}).get("pearson_r", float("nan"))
    if not np.isnan(cum_r) and not np.isnan(spec_r):
        if abs(cum_r) > abs(spec_r):
            logger.info(
                "RESULT: cumulative γ_post scale (r=%.3f) explains K_ℓ better "
                "than σ₁(W_up) (r=%.3f) — confirms the dual-norm theory.",
                cum_r,
                spec_r,
            )
        else:
            logger.info(
                "RESULT: σ₁(W_up) (r=%.3f) still dominates over cumulative γ_post "
                "(r=%.3f) — the dual-norm theory is NOT the primary explanation.",
                spec_r,
                cum_r,
            )

    # -- Cleanup --
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gemma 2 post-norm scale vs K correlation analysis (Exp 09)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device (cuda or cpu). Default: cuda.",
    )
    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip σ₁(W_up) SVD computation (much faster on CPU).",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "models.yml"),
        help="Path to models.yml config.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    logger = _setup_logging()

    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    models_cfg: dict = cfg.get("models", cfg)
    gemma_key = next(
        (k for k in models_cfg if "gemma" in k.lower()),
        None,
    )
    if gemma_key is None:
        raise ValueError(
            "No Gemma model found in models.yml. "
            "Expected a key containing 'gemma'."
        )

    run_analysis(
        model_cfg=models_cfg[gemma_key],
        model_key=gemma_key,
        device=device,
        skip_spectral=args.no_spectral,
        logger=logger,
    )

    logger.info("Experiment 09 complete.")


if __name__ == "__main__":
    main()
