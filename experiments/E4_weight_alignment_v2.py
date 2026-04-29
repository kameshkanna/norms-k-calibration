"""
E4_weight_alignment_v2.py — Weight-space alignment with bootstrap CIs and
random structured direction control.

Upgrades over the original 06_weight_space_alignment.py:
  ✓ Bootstrap B=50 subsamples of N_tr=36 pairs → mean ± std of alignment ratio (M3)
  ✓ Random structured direction control: PCA from NON-contrastive pairs (F4)
  ✓ PC2–5 alignment ratios alongside PC1 (M4)
  ✓ Alignment ratios by layer depth zone to justify middle-50% steering (M5)
  ✓ Proper confidence intervals via percentile bootstrap

Outputs (per model × behavior)
-------------------------------
results/weight_alignment_v2/{model}/{behavior}/
    alignment_bootstrap.csv      — per-bootstrap-sample alignment ratios (B rows)
    alignment_summary.csv        — mean, std, CI95 per PC component per layer zone
    random_control_summary.csv   — same metrics for random structured dirs (F4 control)
    pc_components_alignment.csv  — alignment ratios for PC1–5 per layer (M4)

results/weight_alignment_v2/
    aggregate_summary.csv        — one row per model, mean alignment ratio with CI

Usage
-----
python experiments/E4_weight_alignment_v2.py --model all --behavior all
python experiments/E4_weight_alignment_v2.py --model llama --behavior formality
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ALL_BEHAVIOR_KEYS, ALL_MODEL_KEYS, BEHAVIOR_REGISTRY, DATA_DIR,
    DEVICE, EXPERIMENT_CFG, GLOBAL_SEED, MODEL_REGISTRY, RESULTS_DIR,
    TORCH_DTYPE,
)
from activation_baking.model_utils import ModelInfo, detect_model_info
from activation_baking.extractor import ActivationExtractor
from activation_baking.pca_director import PCADirector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("E4_weight_alignment_v2")

OUT_DIR = RESULTS_DIR / "weight_alignment_v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_contrastive_pairs(behavior_key: str) -> List[Dict]:
    """Load all contrastive pairs for a behavior."""
    path = DATA_DIR / "behaviors" / BEHAVIOR_REGISTRY[behavior_key].data_file
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n")]
    return pairs


def extract_diff_matrix(
    pairs: List[Dict],
    indices: List[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    layer_names: List[str],
    device: str,
) -> Dict[str, np.ndarray]:
    """
    For a subset of pairs (given by indices), return per-layer activation
    difference matrices: Dict[layer_name -> ndarray of shape (n_pairs, hidden)].
    """
    layer_diffs: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}

    for idx in indices:
        pair = pairs[idx]
        pos_text = pair["positive"]
        neg_text = pair["negative"]

        with torch.no_grad():
            pos_acts = extractor.extract(pos_text)
            neg_acts = extractor.extract(neg_text)

        for ln in layer_names:
            if ln in pos_acts and ln in neg_acts:
                diff = (pos_acts[ln] - neg_acts[ln]).float().cpu().numpy()
                layer_diffs[ln].append(diff)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {ln: np.stack(v) for ln, v in layer_diffs.items() if v}


def compute_alignment_ratio(
    directions: np.ndarray,
    singular_vectors: np.ndarray,
    n_random_baseline: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Compute mean-max alignment of `directions` against `singular_vectors`,
    normalized by a random unit vector baseline.

    Args:
        directions:        shape (n_components, d) — unit-normed behavioral directions
        singular_vectors:  shape (top_k, d) — unit-normed right SVs of W_down
        n_random_baseline: number of random unit vectors for baseline
        rng:               numpy Generator for reproducibility

    Returns:
        (mean_max_alignment, random_baseline, alignment_ratio)
    """
    if rng is None:
        rng = np.random.default_rng(GLOBAL_SEED)

    d = directions.shape[1]

    # Absolute cosine similarities: (n_components, top_k)
    dots = np.abs(directions @ singular_vectors.T)

    # Mean-max: for each component, take max alignment over SVs; then average over components
    mean_max = float(dots.max(axis=1).mean())

    # Random baseline: random unit vectors on S^{d-1}
    rand_vecs = rng.standard_normal((n_random_baseline, d))
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    rand_dots = np.abs(rand_vecs @ singular_vectors.T)
    random_baseline = float(rand_dots.max(axis=1).mean())

    ratio = mean_max / (random_baseline + 1e-10)
    return mean_max, random_baseline, ratio


def get_top_singular_vectors(
    weight: torch.Tensor, top_k: int
) -> np.ndarray:
    """
    Compute top-k right singular vectors of a weight matrix via truncated SVD.
    Returns ndarray of shape (top_k, d_in).
    """
    W = weight.float().cpu()
    try:
        _, _, Vt = torch.linalg.svd(W, full_matrices=False)
        vecs = Vt[:top_k].numpy()
    except torch.linalg.LinAlgError:
        # Fallback: numpy SVD
        _, _, Vt = np.linalg.svd(W.numpy(), full_matrices=False)
        vecs = Vt[:top_k]
    # Unit-normalize (should already be, but ensure)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-10)


def run_bootstrap_alignment(
    all_pairs: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    model_info: ModelInfo,
    B: int,
    subsample_size: int,
    top_k: int,
    n_random_baseline: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run B bootstrap subsamples. For each subsample:
      1. Sample subsample_size pairs (with replacement from all_pairs).
      2. Fit PCA on diff matrix.
      3. For each layer: compute alignment of PC1 and PC2–5 against W_down SVs.

    Returns DataFrame with columns:
      bootstrap_idx, layer_idx, layer_name, pc_idx,
      mean_max_alignment, random_baseline, alignment_ratio, depth_zone
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(all_pairs)
    layer_names = model_info.layer_names
    n_layers = len(layer_names)

    records: List[Dict] = []

    for b in tqdm(range(B), desc="Bootstrap subsamples", leave=False):
        # Sample indices with replacement
        indices = rng.integers(0, n_pairs, size=subsample_size).tolist()

        # Extract diff matrix for this subsample
        diff_matrix = extract_diff_matrix(
            all_pairs, indices, model, tokenizer, extractor, layer_names, DEVICE
        )

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name not in diff_matrix:
                continue

            X = diff_matrix[layer_name]  # (subsample_size, d)

            # Fit PCA
            X_centered = X - X.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
                components = Vt[: EXPERIMENT_CFG.n_pca_components]  # (n_components, d)
            except np.linalg.LinAlgError:
                log.warning("SVD failed at layer %s bootstrap %d, skipping.", layer_name, b)
                continue

            # Normalize components
            norms = np.linalg.norm(components, axis=1, keepdims=True)
            components = components / (norms + 1e-10)

            # Get W_down singular vectors for this layer
            try:
                W_down = model_info.get_down_proj(model, layer_name)
                sv = get_top_singular_vectors(W_down, top_k)
            except (AttributeError, KeyError):
                continue

            # Compute alignment for each PC component
            for pc_idx in range(components.shape[0]):
                direction = components[pc_idx : pc_idx + 1]  # (1, d)
                mma, baseline, ratio = compute_alignment_ratio(
                    direction, sv, n_random_baseline=n_random_baseline, rng=rng
                )

                # Depth zone
                depth_frac = layer_idx / max(n_layers - 1, 1)
                if depth_frac < 0.25:
                    zone = "early (0-25%)"
                elif depth_frac < 0.75:
                    zone = "middle (25-75%)"
                else:
                    zone = "late (75-100%)"

                records.append(
                    {
                        "bootstrap_idx": b,
                        "layer_idx": layer_idx,
                        "layer_name": layer_name,
                        "pc_idx": pc_idx,
                        "mean_max_alignment": mma,
                        "random_baseline": baseline,
                        "alignment_ratio": ratio,
                        "depth_zone": zone,
                        "depth_frac": depth_frac,
                    }
                )

    return pd.DataFrame(records)


def run_random_structured_control(
    all_pairs: List[Dict],
    all_behaviors: List[str],
    target_behavior: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    model_info: ModelInfo,
    n_control_dirs: int,
    top_k: int,
    n_random_baseline: int,
    seed: int,
) -> pd.DataFrame:
    """
    F4 control: extract PCA directions from NON-contrastive pairings.

    Strategy: pair positive samples from one behavior with positive samples
    from a DIFFERENT behavior. These have similar structure (both are
    model outputs) but no behavioral contrast — so PCA recovers generic
    activation structure, not behavioral contrast.

    Returns DataFrame with same columns as bootstrap alignment df.
    """
    rng = np.random.default_rng(seed + 999)
    layer_names = model_info.layer_names
    n_layers = len(layer_names)
    records: List[Dict] = []

    # Collect positive samples from all OTHER behaviors
    other_positives: List[str] = []
    for bk in all_behaviors:
        if bk == target_behavior:
            continue
        pairs = load_contrastive_pairs(bk)
        other_positives.extend(p["positive"] for p in pairs)

    # Also collect negatives from the target behavior (same domain, no contrast)
    target_pairs = load_contrastive_pairs(target_behavior)
    target_positives = [p["positive"] for p in target_pairs]

    # Form n_control_dirs "pseudo-pairs" by random pairing of non-contrastive texts
    n_available = min(len(target_positives), len(other_positives))
    shuffle_idx = rng.permutation(n_available)
    pseudo_pairs = [
        (target_positives[i], other_positives[shuffle_idx[i]])
        for i in range(min(n_control_dirs * 2, n_available))
    ]

    # Extract diff matrix for pseudo-pairs
    diff_rows: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}

    for pos_text, neg_text in tqdm(pseudo_pairs[:n_control_dirs], desc="Control dirs", leave=False):
        with torch.no_grad():
            pos_acts = extractor.extract(pos_text)
            neg_acts = extractor.extract(neg_text)
        for ln in layer_names:
            if ln in pos_acts and ln in neg_acts:
                diff = (pos_acts[ln] - neg_acts[ln]).float().cpu().numpy()
                diff_rows[ln].append(diff)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diff_matrix = {ln: np.stack(v) for ln, v in diff_rows.items() if v}

    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name not in diff_matrix:
            continue

        X = diff_matrix[layer_name]
        X_centered = X - X.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            components = Vt[:1]  # just PC1 for control
        except np.linalg.LinAlgError:
            continue

        norms = np.linalg.norm(components, axis=1, keepdims=True)
        components = components / (norms + 1e-10)

        try:
            W_down = model_info.get_down_proj(model, layer_name)
            sv = get_top_singular_vectors(W_down, top_k)
        except (AttributeError, KeyError):
            continue

        mma, baseline, ratio = compute_alignment_ratio(
            components, sv, n_random_baseline=n_random_baseline, rng=rng
        )

        depth_frac = layer_idx / max(n_layers - 1, 1)
        zone = (
            "early (0-25%)" if depth_frac < 0.25
            else "middle (25-75%)" if depth_frac < 0.75
            else "late (75-100%)"
        )

        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "pc_idx": 0,
                "direction_type": "random_structured",
                "mean_max_alignment": mma,
                "random_baseline": baseline,
                "alignment_ratio": ratio,
                "depth_zone": zone,
                "depth_frac": depth_frac,
            }
        )

    return pd.DataFrame(records)


def summarize_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bootstrap results: mean ± std of alignment_ratio per
    (pc_idx, depth_zone). CI95 via 2.5/97.5 percentiles of the bootstrap distribution.
    """
    summary_rows: List[Dict] = []

    for (pc_idx, zone), group in df.groupby(["pc_idx", "depth_zone"]):
        ratios = group["alignment_ratio"].values
        summary_rows.append(
            {
                "pc_idx": pc_idx,
                "depth_zone": zone,
                "mean_ratio": float(ratios.mean()),
                "std_ratio": float(ratios.std()),
                "ci95_lo": float(np.percentile(ratios, 2.5)),
                "ci95_hi": float(np.percentile(ratios, 97.5)),
                "n_bootstrap": len(ratios),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["pc_idx", "depth_zone"])


# ---------------------------------------------------------------------------
# Main per-model-behavior runner
# ---------------------------------------------------------------------------

def run_model_behavior(
    model_key: str,
    behavior_key: str,
    args: argparse.Namespace,
) -> None:
    cfg = MODEL_REGISTRY[model_key]
    out_dir = OUT_DIR / model_key / behavior_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (out_dir / "alignment_summary.csv").exists() and not args.overwrite:
        log.info("Skipping %s/%s (already done).", model_key, behavior_key)
        return

    log.info("=== %s / %s ===", cfg.label, behavior_key)

    # Load model
    log.info("Loading model: %s", cfg.hf_id)
    dtype = getattr(torch, TORCH_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    model_info: ModelInfo = detect_model_info(model, tokenizer)
    extractor = ActivationExtractor(model, tokenizer, model_info, device=DEVICE)

    # Load contrastive pairs
    all_pairs = load_contrastive_pairs(behavior_key)
    log.info("Loaded %d pairs for %s.", len(all_pairs), behavior_key)

    # ── Bootstrap alignment ──────────────────────────────────────────────────
    log.info("Running B=%d bootstrap subsamples...", EXPERIMENT_CFG.n_bootstrap_subsets)
    bootstrap_df = run_bootstrap_alignment(
        all_pairs=all_pairs,
        model=model,
        tokenizer=tokenizer,
        extractor=extractor,
        model_info=model_info,
        B=EXPERIMENT_CFG.n_bootstrap_subsets,
        subsample_size=EXPERIMENT_CFG.bootstrap_subsample_size,
        top_k=EXPERIMENT_CFG.top_k_singular_vectors,
        n_random_baseline=EXPERIMENT_CFG.n_random_baseline_dirs,
        seed=GLOBAL_SEED,
    )
    bootstrap_df.to_csv(out_dir / "alignment_bootstrap.csv", index=False)
    log.info("Saved bootstrap data: %d rows.", len(bootstrap_df))

    summary_df = summarize_bootstrap(bootstrap_df)
    summary_df.to_csv(out_dir / "alignment_summary.csv", index=False)

    # Log key result: PC1 alignment in middle 25-75% zone
    pc1_mid = summary_df[
        (summary_df["pc_idx"] == 0) & (summary_df["depth_zone"] == "middle (25-75%)")
    ]
    if len(pc1_mid) > 0:
        r = pc1_mid.iloc[0]
        log.info(
            "PC1 alignment (middle layers): %.3f ± %.3f [CI95: %.3f–%.3f]",
            r["mean_ratio"], r["std_ratio"], r["ci95_lo"], r["ci95_hi"],
        )

    # ── Random structured direction control (F4) ─────────────────────────────
    log.info("Running random structured direction control (F4)...")
    control_df = run_random_structured_control(
        all_pairs=all_pairs,
        all_behaviors=list(ALL_BEHAVIOR_KEYS),
        target_behavior=behavior_key,
        model=model,
        tokenizer=tokenizer,
        extractor=extractor,
        model_info=model_info,
        n_control_dirs=EXPERIMENT_CFG.n_random_structured_dirs,
        top_k=EXPERIMENT_CFG.top_k_singular_vectors,
        n_random_baseline=EXPERIMENT_CFG.n_random_baseline_dirs,
        seed=GLOBAL_SEED,
    )
    control_df.to_csv(out_dir / "random_control_summary.csv", index=False)

    # Log control vs behavioral comparison
    ctrl_mid = control_df[control_df["depth_zone"] == "middle (25-75%)"]["alignment_ratio"]
    beh_mid = bootstrap_df[
        (bootstrap_df["pc_idx"] == 0) & (bootstrap_df["depth_zone"] == "middle (25-75%)")
    ]["alignment_ratio"]
    if len(ctrl_mid) > 0 and len(beh_mid) > 0:
        log.info(
            "Behavioral PC1 vs random structured: %.3f vs %.3f (advantage: %.3f)",
            beh_mid.mean(), ctrl_mid.mean(), beh_mid.mean() - ctrl_mid.mean(),
        )

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_aggregate_summary(model_keys: List[str], behavior_keys: List[str]) -> None:
    """Combine all model-behavior summaries into one aggregate CSV."""
    records: List[Dict] = []

    for model_key in model_keys:
        cfg = MODEL_REGISTRY[model_key]
        for behavior_key in behavior_keys:
            summary_path = OUT_DIR / model_key / behavior_key / "alignment_summary.csv"
            control_path = OUT_DIR / model_key / behavior_key / "random_control_summary.csv"

            if not summary_path.exists():
                continue

            summary = pd.read_csv(summary_path)
            # PC1, middle layers
            pc1_mid = summary[
                (summary["pc_idx"] == 0) & (summary["depth_zone"] == "middle (25-75%)")
            ]
            if len(pc1_mid) == 0:
                continue

            r = pc1_mid.iloc[0]
            rec = {
                "model": cfg.label,
                "model_key": model_key,
                "behavior": behavior_key,
                "pc1_ratio_mean": r["mean_ratio"],
                "pc1_ratio_std": r["std_ratio"],
                "pc1_ci95_lo": r["ci95_lo"],
                "pc1_ci95_hi": r["ci95_hi"],
            }

            if control_path.exists():
                ctrl = pd.read_csv(control_path)
                ctrl_mid = ctrl[ctrl["depth_zone"] == "middle (25-75%)"]["alignment_ratio"]
                rec["random_structured_ratio"] = float(ctrl_mid.mean()) if len(ctrl_mid) > 0 else float("nan")
                rec["behavioral_advantage"] = rec["pc1_ratio_mean"] - rec.get("random_structured_ratio", 0)

            records.append(rec)

    if records:
        agg = pd.DataFrame(records)
        agg.to_csv(OUT_DIR / "aggregate_summary.csv", index=False)
        log.info("Saved aggregate summary: %d rows.", len(agg))

        # Print clean summary table
        log.info("\n=== AGGREGATE RESULTS ===")
        for model_key in model_keys:
            model_rows = agg[agg["model_key"] == model_key]
            if len(model_rows) == 0:
                continue
            mean_ratio = model_rows["pc1_ratio_mean"].mean()
            mean_ci = (model_rows["pc1_ci95_hi"] - model_rows["pc1_ci95_lo"]).mean() / 2
            log.info(
                "  %-15s: %.3f ± %.3f (mean ± half-CI95 across behaviors)",
                MODEL_REGISTRY[model_key].label, mean_ratio, mean_ci,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight alignment v2 with bootstrap CIs")
    p.add_argument("--model", default="all", help="Model key or 'all'")
    p.add_argument("--behavior", default="all", help="Behavior key or 'all'")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip experiments, just rebuild aggregate summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_keys = ALL_MODEL_KEYS if args.model == "all" else [args.model]
    behavior_keys = ALL_BEHAVIOR_KEYS if args.behavior == "all" else [args.behavior]

    if not args.aggregate_only:
        for model_key in model_keys:
            for behavior_key in behavior_keys:
                run_model_behavior(model_key, behavior_key, args)

    build_aggregate_summary(list(model_keys), list(behavior_keys))


if __name__ == "__main__":
    main()
