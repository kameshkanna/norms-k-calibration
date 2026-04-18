"""
experiments/03_k_calibration_validation.py

Validate the K-calibration formula K_i = μ_i / √hidden_size by comparing the
computed K-values against spectral norms of weight matrices at each layer.

This is the primary empirical support for **Claim 2** of the paper:
    *The K formula has a spectral-geometry interpretation — K_i tracks the
    operator norm of the dominant weight matrices at each layer.*

For each architecture the script:
  1. Loads (or recomputes) per-layer mean norms from script 01's output.
  2. Computes K-values: K_i = mean_norm_i / √hidden_size.
  3. Uses KCalibrator to compute spectral norms of down_proj, up_proj, and
     o_proj weight matrices at each layer.
  4. Computes Pearson and Spearman correlations between K-values and spectral
     norms for each weight type.
  5. Saves per-layer comparison CSV and per-model correlation JSON.

Outputs
-------
{output_dir}/{model}_k_vs_spectral.csv
    Columns: layer_idx, layer_name, mean_norm, k_value, spectral_norm_down,
    spectral_norm_up, spectral_norm_o, k_spectral_ratio_down,
    k_spectral_ratio_up, k_spectral_ratio_o.
{output_dir}/{model}_correlation.json
    Keys: pearson_r_{weight_type}, spearman_r_{weight_type},
    p_value_{weight_type} for each weight type.

Usage
-----
python experiments/03_k_calibration_validation.py --model all --device cuda
python experiments/03_k_calibration_validation.py --model llama --weight-type down_proj
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import gc
import json
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as scipy_stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed

from activation_baking.model_utils import ModelInfo, detect_model_info, get_layer_module
from activation_baking.calibrator import KCalibrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_TYPE_CHOICES: List[str] = ["down_proj", "up_proj", "o_proj", "all"]

# Maps CLI weight-type key → arch_patterns key (see model_utils._ARCH_PATTERNS)
_WEIGHT_TYPE_TO_PATTERN_KEY: Dict[str, str] = {
    "down_proj": "mlp_down_proj",
    "up_proj": "mlp_up_proj",
    "o_proj": "attn_o_proj",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(tag: str, output_dir: Path) -> None:
    """Configure root logger to write to console and a timestamped file.

    Args:
        tag: Short identifier used in the log filename.
        output_dir: Base output directory; logs go under ``results/logs/``.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"03_k_calibration_validation_{tag}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("Log file: %s", log_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _set_global_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string with graceful CUDA fallback.

    Args:
        device_str: Requested device string.

    Returns:
        Validated torch.device.
    """
    logger = logging.getLogger(__name__)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _gpu_mem_gb(device: torch.device) -> float:
    """Return current GPU allocation in GiB or 0.0 for CPU.

    Args:
        device: The torch device to query.

    Returns:
        Allocated memory in GiB.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Norm profile loading / fallback
# ---------------------------------------------------------------------------


def _load_norm_profile(
    model_key: str,
    norm_profiles_dir: Path,
    model_cfg: Dict,
    device: torch.device,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load the norm profile CSV for a model, recomputing it if absent.

    The norm profile is the output of ``experiments/01_norm_profiling.py``.
    If the CSV does not exist this function re-invokes script 01 as a subprocess
    to generate it before loading.

    Args:
        model_key: Short model key (e.g. ``"llama"``).
        norm_profiles_dir: Directory where ``{model_key}.csv`` is expected.
        model_cfg: Config sub-dict for the model (unused if CSV already exists).
        device: Device string forwarded when re-running script 01.
        seed: Seed forwarded when re-running script 01.
        logger: Logger for status messages.

    Returns:
        DataFrame with at minimum columns: layer_idx, layer_name, mean_norm,
        std_norm, k_value, hidden_size, architecture.

    Raises:
        RuntimeError: If the CSV cannot be found or regenerated.
    """
    csv_path = norm_profiles_dir / f"{model_key}.csv"
    if csv_path.exists():
        logger.info("Loading norm profile from %s", csv_path)
        return pd.read_csv(csv_path)

    logger.warning(
        "Norm profile not found at %s.  Running 01_norm_profiling.py inline…",
        csv_path,
    )
    script_path = Path("experiments/01_norm_profiling.py")
    if not script_path.exists():
        raise RuntimeError(
            f"Cannot regenerate norm profile: {script_path} not found."
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--model", model_key,
        "--device", str(device),
        "--output-dir", str(norm_profiles_dir),
        "--seed", str(seed),
    ]
    logger.info("Subprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"01_norm_profiling.py exited with code {result.returncode} "
            f"for model '{model_key}'."
        )

    if not csv_path.exists():
        raise RuntimeError(
            f"01_norm_profiling.py ran but {csv_path} was not produced."
        )

    logger.info("Norm profile regenerated → %s", csv_path)
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Spectral norm computation
# ---------------------------------------------------------------------------


def _compute_spectral_norm_matrix(weight: torch.Tensor) -> float:
    """Compute the spectral (operator-2) norm of a 2-D weight matrix.

    The spectral norm equals the largest singular value σ_max.  It bounds how
    much the linear map can stretch any input vector and is the quantity most
    directly comparable to the K-value formula under our geometric interpretation.

    Args:
        weight: 2-D float tensor of shape [out_features, in_features].

    Returns:
        Spectral norm as a Python float.

    Raises:
        ValueError: If weight is not 2-D.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"Expected 2-D weight tensor, got shape {tuple(weight.shape)}."
        )
    # Move to CPU in float32 for numerically stable SVD
    w = weight.detach().float().cpu()
    # torch.linalg.matrix_norm with ord=2 computes σ_max via SVD
    return torch.linalg.matrix_norm(w, ord=2).item()


def _collect_spectral_norms(
    model: AutoModelForCausalLM,
    model_info: ModelInfo,
    weight_types: List[str],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    """Compute spectral norms for all requested weight types at every layer.

    Args:
        model: Loaded eval-mode causal LM.
        model_info: Structural metadata for the model.
        weight_types: List of weight-type keys to compute
            (subset of ``["down_proj", "up_proj", "o_proj"]``).
        device: Device on which the model lives (used for logging only).
        logger: Logger for progress messages.

    Returns:
        Dict mapping ``layer_name → {weight_type: spectral_norm}``.
    """
    # Use KCalibrator for spectral norms; fall back to direct computation if needed.
    calibrator = KCalibrator()

    results: Dict[str, Dict[str, float]] = {
        ln: {} for ln in model_info.layer_module_names
    }

    for weight_type in tqdm(
        weight_types, desc="Weight types", unit="wt", dynamic_ncols=True, leave=False
    ):
        pattern_key = _WEIGHT_TYPE_TO_PATTERN_KEY[weight_type]
        arch_patterns = model_info.arch_patterns

        logger.info("Computing spectral norms for weight type '%s'…", weight_type)

        for layer_idx, layer_name in enumerate(
            tqdm(
                model_info.layer_module_names,
                desc=f"  Layers ({weight_type})",
                unit="layer",
                dynamic_ncols=True,
                leave=False,
            )
        ):
            sub_path = arch_patterns.get(pattern_key, "")
            if not sub_path:
                logger.warning(
                    "Pattern key '%s' not in arch_patterns for layer '%s'; skipping.",
                    pattern_key,
                    layer_name,
                )
                results[layer_name][weight_type] = float("nan")
                continue

            full_path = f"{layer_name}.{sub_path}"
            module = get_layer_module(model, full_path)
            if not hasattr(module, "weight"):
                logger.warning(
                    "Module '%s' has no .weight attribute; spectral norm = NaN.",
                    full_path,
                )
                results[layer_name][weight_type] = float("nan")
                continue

            spectral_norm = calibrator.compute_spectral_norm(module.weight.data)
            results[layer_name][weight_type] = spectral_norm

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def _compute_correlations(
    k_values: np.ndarray,
    spectral_norms: np.ndarray,
    weight_type: str,
) -> Dict[str, float]:
    """Compute Pearson and Spearman correlations between K-values and spectral norms.

    Layers with NaN spectral norms are excluded before computing statistics.

    Args:
        k_values: Array of per-layer K-values, shape [num_layers].
        spectral_norms: Array of per-layer spectral norms, shape [num_layers].
        weight_type: Label for logging.

    Returns:
        Dict with keys: pearson_r, spearman_r, pearson_p, spearman_p, n_layers_used.
    """
    logger = logging.getLogger(__name__)
    mask = ~(np.isnan(k_values) | np.isnan(spectral_norms))
    kv = k_values[mask]
    sn = spectral_norms[mask]

    if len(kv) < 3:
        logger.warning(
            "Fewer than 3 valid data points for '%s' correlation; returning NaN.",
            weight_type,
        )
        return {
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_p": float("nan"),
            "n_layers_used": int(mask.sum()),
        }

    pearson_r, pearson_p = scipy_stats.pearsonr(kv, sn)
    spearman_r, spearman_p = scipy_stats.spearmanr(kv, sn)

    logger.info(
        "Correlation [%s]: pearson_r=%.4f (p=%.3e)  spearman_r=%.4f (p=%.3e)  n=%d",
        weight_type,
        pearson_r,
        pearson_p,
        spearman_r,
        spearman_p,
        int(mask.sum()),
    )
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "pearson_p": float(pearson_p),
        "spearman_p": float(spearman_p),
        "n_layers_used": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def _validate_single_model(
    model_key: str,
    model_cfg: Dict,
    weight_types: List[str],
    device: torch.device,
    output_dir: Path,
    norm_profiles_dir: Path,
    seed: int,
) -> None:
    """Run K-vs-spectral validation for one architecture.

    Args:
        model_key: Short architecture key.
        model_cfg: Config sub-dict from ``models.yml``.
        weight_types: List of weight-type keys to evaluate.
        device: Target torch.device.
        output_dir: Root directory for output CSVs and JSONs.
        norm_profiles_dir: Directory containing norm profile CSVs.
        seed: Random seed.
    """
    logger = logging.getLogger(__name__)
    hf_id: str = model_cfg["huggingface_id"]
    hidden_size: int = model_cfg["hidden_size"]
    sqrt_hidden: float = math.sqrt(hidden_size)

    t_start = time.time()

    # ---- Load norm profile ----
    norm_df = _load_norm_profile(
        model_key=model_key,
        norm_profiles_dir=norm_profiles_dir,
        model_cfg=model_cfg,
        device=device,
        seed=seed,
        logger=logger,
    )

    # Validate required columns
    required_cols = {"layer_idx", "layer_name", "mean_norm"}
    missing_cols = required_cols - set(norm_df.columns)
    if missing_cols:
        raise ValueError(
            f"Norm profile CSV for '{model_key}' is missing columns: {missing_cols}"
        )

    # Recompute K-values from mean_norm (to ensure formula consistency even if
    # the CSV was generated with a different hidden_size value).
    norm_df = norm_df.sort_values("layer_idx").reset_index(drop=True)
    norm_df["k_value"] = norm_df["mean_norm"] / sqrt_hidden

    logger.info(
        "Norm profile loaded: %d layers  mean_k=%.6f",
        len(norm_df),
        norm_df["k_value"].mean(),
    )

    # ---- Load model ----
    logger.info("Loading model for spectral norm computation: %s", hf_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_info: ModelInfo = detect_model_info(model, hf_id)

    logger.info("GPU mem after load: %.2f GiB", _gpu_mem_gb(device))

    # ---- Compute spectral norms ----
    spectral_norms_by_layer = _collect_spectral_norms(
        model=model,
        model_info=model_info,
        weight_types=weight_types,
        device=device,
        logger=logger,
    )

    # ---- Build output DataFrame ----
    k_arr = norm_df["k_value"].values
    records = []
    for _, row in norm_df.iterrows():
        layer_name = row["layer_idx"], row["layer_name"]
        rec = {
            "layer_idx": int(row["layer_idx"]),
            "layer_name": row["layer_name"],
            "mean_norm": float(row["mean_norm"]),
            "k_value": float(row["k_value"]),
        }
        sn_dict = spectral_norms_by_layer.get(row["layer_name"], {})
        for wt in ["down_proj", "up_proj", "o_proj"]:
            sn = sn_dict.get(wt, float("nan"))
            rec[f"spectral_norm_{wt}"] = sn
            rec[f"k_spectral_ratio_{wt}"] = (
                float(row["k_value"]) / sn if (not math.isnan(sn) and sn > 0) else float("nan")
            )
        records.append(rec)

    result_df = pd.DataFrame(records)

    # ---- Correlation analysis ----
    correlation_report: Dict[str, object] = {
        "model_key": model_key,
        "hf_id": hf_id,
        "architecture": model_cfg["architecture"],
        "hidden_size": hidden_size,
        "n_layers": len(result_df),
        "weight_types_evaluated": weight_types,
    }

    for wt in weight_types:
        col = f"spectral_norm_{wt}"
        if col not in result_df.columns:
            continue
        corr = _compute_correlations(
            k_values=result_df["k_value"].values,
            spectral_norms=result_df[col].values,
            weight_type=wt,
        )
        for stat_key, stat_val in corr.items():
            correlation_report[f"{stat_key}_{wt}"] = stat_val

        logger.info(
            "K-spectral correlation for %s [%s]: pearson_r=%.3f  spearman_r=%.3f",
            model_key,
            wt,
            corr.get("pearson_r", float("nan")),
            corr.get("spearman_r", float("nan")),
        )

    # ---- Persist outputs ----
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{model_key}_k_vs_spectral.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info("K-vs-spectral CSV saved → %s", csv_path)

    pt_path = output_dir / f"{model_key}_k_vs_spectral.pt"
    torch.save(
        {
            "result_df_records": result_df.to_dict(orient="records"),
            "spectral_norms_by_layer": spectral_norms_by_layer,
            "model_key": model_key,
            "seed": seed,
        },
        str(pt_path),
    )
    logger.info(".pt checkpoint saved → %s", pt_path)

    corr_path = output_dir / f"{model_key}_correlation.json"
    with corr_path.open("w", encoding="utf-8") as fh:
        json.dump(correlation_report, fh, indent=2)
    logger.info("Correlation JSON saved → %s", corr_path)

    elapsed = time.time() - t_start
    logger.info(
        "Model '%s' done in %.1fs  GPU mem=%.2f GiB",
        model_key,
        elapsed,
        _gpu_mem_gb(device),
    )

    # ---- Cleanup ----
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Fully configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate K = μ/√hidden against spectral norms of weight matrices. "
            "Primary support for Claim 2 of the paper: K-formula has a spectral "
            "geometry interpretation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "qwen", "gemma", "mistral", "all"],
        default="all",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/k_calibration",
        dest="output_dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=WEIGHT_TYPE_CHOICES,
        default="all",
        dest="weight_type",
        help=(
            "Weight matrix type(s) to compute spectral norms for. "
            "'all' evaluates down_proj, up_proj, and o_proj."
        ),
    )
    parser.add_argument(
        "--norm-profiles-dir",
        type=str,
        default="results/norm_profiles",
        dest="norm_profiles_dir",
        help="Directory containing norm profile CSVs from script 01.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for K-calibration validation experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    norm_profiles_dir = Path(args.norm_profiles_dir)
    _setup_logging(args.model, output_dir)
    logger = logging.getLogger(__name__)

    device = _resolve_device(args.device)
    _set_global_seed(args.seed)

    # ---- Load models config ----
    cfg_path = Path("config/models.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Models config not found: {cfg_path.resolve()}")
    with cfg_path.open() as fh:
        models_cfg: Dict = yaml.safe_load(fh)

    all_model_keys = list(models_cfg["models"].keys())
    target_keys: List[str] = all_model_keys if args.model == "all" else [args.model]

    # Resolve weight types
    if args.weight_type == "all":
        weight_types = list(_WEIGHT_TYPE_TO_PATTERN_KEY.keys())
    else:
        weight_types = [args.weight_type]

    logger.info(
        "K-calibration validation: models=%s  weight_types=%s  device=%s  seed=%d",
        target_keys,
        weight_types,
        device,
        args.seed,
    )

    for model_key in tqdm(target_keys, desc="Models", unit="model", dynamic_ncols=True):
        logger.info("=" * 72)
        logger.info("MODEL: %s", model_key)
        logger.info("=" * 72)
        _validate_single_model(
            model_key=model_key,
            model_cfg=models_cfg["models"][model_key],
            weight_types=weight_types,
            device=device,
            output_dir=output_dir,
            norm_profiles_dir=norm_profiles_dir,
            seed=args.seed,
        )

    logger.info("K-calibration validation complete for models: %s", target_keys)


if __name__ == "__main__":
    main()
