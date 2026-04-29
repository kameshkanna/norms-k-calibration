"""
E7_k_sensitivity_curve.py — K sensitivity sweep.

Addresses:
  M2: Shows that the calibrated K sits near a "natural optimum" by sweeping
      κ × K_l (where κ ∈ {0.01, ..., 50}) and measuring directional fidelity
      at each scale.
  M3: 95% CI bands from bootstrap over prompt subsets.

Protocol:
  For the formality behavior (most reliable signal, per ExperimentConfig):
    For each model × each κ multiplier (15 values, log-spaced):
      1. Steer with α_l = κ × K_l at each steering layer.
      2. Measure directional fidelity = mean cosine shift in target direction.
      3. Also measure: mean activation L2 norm after steering (to detect blowup).
    κ=1.0 is our calibrated value → report its fidelity and rank.
    Bootstrap CI: B=50 subsamples of n=36 prompt pairs.

Outputs:
  results/k_sensitivity/{model}/sensitivity_curve.csv
  results/k_sensitivity/aggregate_sensitivity.csv
  figures/fig_k_sensitivity.pdf
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_MODEL_KEYS,
    BEHAVIOR_REGISTRY,
    DATA_DIR,
    EXPERIMENT_CFG,
    GLOBAL_SEED,
    MODEL_REGISTRY,
    RESULTS_DIR,
    TORCH_DTYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "k_sensitivity"
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
BEHAVIOR_KEY = EXPERIMENT_CFG.k_sensitivity_behavior  # "formality"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_contrastive_pairs(behavior_key: str) -> List[Tuple[str, str]]:
    bcfg = BEHAVIOR_REGISTRY[behavior_key]
    path = DATA_DIR / "behaviors" / bcfg.data_file
    if not path.exists():
        raise FileNotFoundError(f"Behavior data not found: {path}")
    pairs: List[Tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["positive"], obj["negative"]))
    return pairs


# ---------------------------------------------------------------------------
# Activation utilities
# ---------------------------------------------------------------------------

def get_steering_layers(num_layers: int) -> List[int]:
    start = int(EXPERIMENT_CFG.steering_layer_start_frac * num_layers)
    end = int(EXPERIMENT_CFG.steering_layer_end_frac * num_layers)
    return list(range(start, end))


def extract_mean_activations(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    layer_indices: List[int],
    device: torch.device,
    max_length: int = 128,
) -> Dict[int, np.ndarray]:
    """
    Returns mean-pooled residual stream activations for a batch of texts.
    Dict[layer_idx → ndarray(n_texts, hidden)].
    """
    captured: Dict[int, List[np.ndarray]] = {l: [] for l in layer_indices}

    def make_hook(l: int):
        def fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[l].append(h.detach().float().cpu().numpy()[0].mean(axis=0))
        return fn

    hooks = []
    for l in layer_indices:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    return {l: np.stack(captured[l], axis=0) for l in layer_indices if captured[l]}


def compute_k_calibration(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    num_layers: int,
    hidden_size: int,
    device: torch.device,
) -> np.ndarray:
    """K_l = μ̄_l / √d. Returns array (num_layers,)."""
    layer_indices = list(range(num_layers))
    acts = extract_mean_activations(model, tokenizer, prompts, layer_indices, device)
    norms = np.array([acts[l].mean(axis=0) for l in range(num_layers) if l in acts])
    # acts[l] shape: (n_prompts, hidden) → L2 norm per prompt, then mean
    mean_norms = np.array([
        np.linalg.norm(acts[l], axis=1).mean() if l in acts else 0.0
        for l in range(num_layers)
    ])
    return mean_norms / np.sqrt(hidden_size)


def extract_pca_direction(
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    steering_layers: List[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """PC1 of diff matrix at each steering layer."""
    pos_texts = [p for p, _ in pairs]
    neg_texts = [n for _, n in pairs]
    pos_acts = extract_mean_activations(model, tokenizer, pos_texts, steering_layers, device)
    neg_acts = extract_mean_activations(model, tokenizer, neg_texts, steering_layers, device)

    directions: Dict[int, np.ndarray] = {}
    for l in steering_layers:
        if l not in pos_acts or l not in neg_acts:
            continue
        n = min(len(pos_acts[l]), len(neg_acts[l]))
        diff = pos_acts[l][:n] - neg_acts[l][:n]
        pca = PCA(n_components=1)
        pca.fit(diff)
        directions[l] = pca.components_[0]
    return directions


# ---------------------------------------------------------------------------
# Steered forward pass + fidelity measurement
# ---------------------------------------------------------------------------

class SteeringHook:
    """
    Adds κ × K_l × direction to the residual stream at each steering layer.
    """

    def __init__(
        self,
        model: nn.Module,
        directions: Dict[int, np.ndarray],
        alpha_per_layer: Dict[int, float],
    ) -> None:
        self.model = model
        self.directions = {
            l: torch.from_numpy(d).float() for l, d in directions.items()
        }
        self.alpha_per_layer = alpha_per_layer
        self._hooks: list = []
        self.post_norms: Dict[int, float] = {}

    def _make_hook(self, l: int):
        def fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if l in self.directions:
                vec = self.directions[l].to(h.device, h.dtype)
                alpha = self.alpha_per_layer.get(l, 0.0)
                h = h + alpha * vec
            self.post_norms[l] = float(h.float().norm(dim=-1).mean().item())
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return fn

    def __enter__(self):
        for l in self.directions:
            h = self.model.model.layers[l].register_forward_hook(self._make_hook(l))
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def measure_directional_fidelity(
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    directions: Dict[int, np.ndarray],
    alpha_per_layer: Dict[int, float],
    steering_layers: List[int],
    device: torch.device,
    max_length: int = 128,
) -> Tuple[float, float]:
    """
    Directional fidelity = mean cosine similarity between the steering direction
    and the actual activation shift caused by steering.

    Also returns mean post-steering activation norm.
    Returns: (mean_fidelity, mean_post_norm).
    """
    fidelities: List[float] = []
    post_norms_list: List[float] = []

    for pos_text, _ in pairs:
        inputs = tokenizer(pos_text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)

        # Unsteered activation
        baseline_acts: Dict[int, np.ndarray] = {}
        def make_baseline_hook(l: int):
            def fn(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                baseline_acts[l] = h.detach().float().cpu().numpy()[0].mean(axis=0)
            return fn

        baseline_hooks = []
        for l in steering_layers:
            bh = model.model.layers[l].register_forward_hook(make_baseline_hook(l))
            baseline_hooks.append(bh)
        with torch.no_grad():
            model(**inputs)
        for bh in baseline_hooks:
            bh.remove()

        # Steered activation
        steered_acts: Dict[int, np.ndarray] = {}
        def make_steer_hook(l: int):
            def fn(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if l in directions:
                    vec = torch.from_numpy(directions[l]).float().to(h.device, h.dtype)
                    alpha = alpha_per_layer.get(l, 0.0)
                    h = h + alpha * vec
                steered_acts[l] = h.detach().float().cpu().numpy()[0].mean(axis=0)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return fn

        steer_hooks = []
        for l in steering_layers:
            sh = model.model.layers[l].register_forward_hook(make_steer_hook(l))
            steer_hooks.append(sh)
        with torch.no_grad():
            model(**inputs)
        for sh in steer_hooks:
            sh.remove()

        # Fidelity per layer
        for l in steering_layers:
            if l not in baseline_acts or l not in steered_acts or l not in directions:
                continue
            delta = steered_acts[l] - baseline_acts[l]
            d = directions[l]
            cos = float(np.dot(delta, d) / (np.linalg.norm(delta) * np.linalg.norm(d) + 1e-10))
            fidelities.append(cos)
            post_norms_list.append(float(np.linalg.norm(steered_acts[l])))

    mean_fid = float(np.mean(fidelities)) if fidelities else 0.0
    mean_norm = float(np.mean(post_norms_list)) if post_norms_list else 0.0
    return mean_fid, mean_norm


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_fidelity(
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    directions: Dict[int, np.ndarray],
    k_values: np.ndarray,
    kappa: float,
    steering_layers: List[int],
    device: torch.device,
    B: int,
    subsample_size: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Bootstrap percentile CI on directional fidelity at a given κ.
    Returns (mean, ci_lo, ci_hi).
    """
    alpha_per_layer = {l: kappa * float(k_values[l]) for l in steering_layers}
    bootstrap_means: List[float] = []

    for _ in range(B):
        indices = rng.choice(len(pairs), size=min(subsample_size, len(pairs)), replace=True)
        subset = [pairs[i] for i in indices]
        fid, _ = measure_directional_fidelity(
            subset, model, tokenizer, directions, alpha_per_layer, steering_layers, device
        )
        bootstrap_means.append(fid)

    arr = np.array(bootstrap_means)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_k_sensitivity(
    model_key: str,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    K sensitivity sweep for one model over the formality behavior.
    Returns DataFrame with columns: kappa, mean_fidelity, ci_lo, ci_hi, mean_post_norm.
    """
    out_dir = OUT_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "sensitivity_curve.csv"

    if curve_path.exists() and not overwrite:
        log.info("Checkpoint found — skipping %s", model_key)
        return pd.read_csv(curve_path)

    mcfg = MODEL_REGISTRY[model_key]
    num_layers = mcfg.num_layers
    steering_layers = get_steering_layers(num_layers)

    log.info("  Loading contrastive pairs...")
    try:
        pairs = load_contrastive_pairs(BEHAVIOR_KEY)
    except FileNotFoundError as e:
        log.warning("  %s — skipping.", e)
        return pd.DataFrame()

    log.info("  Computing K-calibration values...")
    from config import GENERATION_EVAL_PROMPTS
    k_values = compute_k_calibration(
        model, tokenizer,
        GENERATION_EVAL_PROMPTS[:EXPERIMENT_CFG.n_calibration_prompts],
        num_layers, mcfg.hidden_size, device,
    )

    log.info("  Extracting PCA behavioral directions...")
    directions = extract_pca_direction(pairs, model, tokenizer, steering_layers, device)
    log.info("  Got directions for %d layers.", len(directions))

    rng = np.random.default_rng(GLOBAL_SEED)
    records: List[dict] = []

    for kappa in tqdm(EXPERIMENT_CFG.k_multipliers, desc=f"K sweep [{model_key}]"):
        alpha_per_layer = {l: kappa * float(k_values[l]) for l in steering_layers}
        mean_fid, mean_norm = measure_directional_fidelity(
            pairs, model, tokenizer, directions, alpha_per_layer, steering_layers, device
        )
        mean_bs, ci_lo, ci_hi = bootstrap_fidelity(
            pairs, model, tokenizer, directions, k_values, kappa,
            steering_layers, device,
            B=EXPERIMENT_CFG.n_bootstrap_subsets,
            subsample_size=EXPERIMENT_CFG.bootstrap_subsample_size,
            rng=rng,
        )
        records.append({
            "kappa": kappa,
            "mean_fidelity": mean_fid,
            "bootstrap_mean": mean_bs,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "mean_post_norm": mean_norm,
            "is_calibrated": abs(kappa - 1.0) < 1e-6,
        })
        log.info("  κ=%.2f | fidelity=%.4f [%.4f, %.4f] | post_norm=%.2f",
                 kappa, mean_fid, ci_lo, ci_hi, mean_norm)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(curve_path, index=False)
    log.info("Saved: %s", curve_path)
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_k_sensitivity(model_keys: List[str], out_path: Path) -> None:
    """
    Line plot: x=κ (log scale), y=directional fidelity, shaded 95% CI,
    vertical line at κ=1.0 (calibrated). One line per model.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping figure.")
        return

    colors = {"llama": "#4e79a7", "qwen": "#f28e2b", "mistral": "#59a14f", "gemma": "#e15759"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for mk in model_keys:
        path = OUT_DIR / mk / "sensitivity_curve.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("kappa")
        label = MODEL_REGISTRY[mk].label if mk in MODEL_REGISTRY else mk
        color = colors.get(mk, "#888888")
        ax.plot(df["kappa"], df["mean_fidelity"], marker="o", markersize=4,
                label=label, color=color, linewidth=1.8)
        ax.fill_between(df["kappa"], df["ci_lo"], df["ci_hi"],
                        alpha=0.15, color=color)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.2,
               label=r"$\kappa=1$ (calibrated $K$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ (multiplier of calibrated $K_\ell$)", fontsize=12)
    ax.set_ylabel("Directional Fidelity (mean cosine shift)", fontsize=12)
    ax.set_title(f"K Sensitivity Curve — {BEHAVIOR_KEY.replace('_', ' ').title()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


def plot_norm_blowup(model_keys: List[str], out_path: Path) -> None:
    """Post-steering activation norm vs κ — shows blowup at large κ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"llama": "#4e79a7", "qwen": "#f28e2b", "mistral": "#59a14f", "gemma": "#e15759"}
    fig, ax = plt.subplots(figsize=(7, 4))

    for mk in model_keys:
        path = OUT_DIR / mk / "sensitivity_curve.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("kappa")
        color = colors.get(mk, "#888888")
        label = MODEL_REGISTRY[mk].label if mk in MODEL_REGISTRY else mk
        ax.plot(df["kappa"], df["mean_post_norm"], marker="s", markersize=3,
                label=label, color=color, linewidth=1.5)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$", fontsize=12)
    ax.set_ylabel("Post-steering L2 norm", fontsize=12)
    ax.set_title("Activation Norm After Steering", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def build_aggregate_sensitivity(model_keys: List[str]) -> pd.DataFrame:
    """Merges per-model CSVs and adds rank of κ=1.0 by fidelity."""
    records: List[dict] = []
    for mk in model_keys:
        path = OUT_DIR / mk / "sensitivity_curve.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("mean_fidelity", ascending=False).reset_index(drop=True)
        df["rank_by_fidelity"] = df.index + 1
        df["model"] = mk
        records.append(df)

    if not records:
        return pd.DataFrame()

    agg = pd.concat(records, ignore_index=True)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E7: K sensitivity curve")
    p.add_argument("--model", choices=list(ALL_MODEL_KEYS) + ["all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = list(ALL_MODEL_KEYS) if args.model == "all" else [args.model]
    device = torch.device(args.device)
    dtype = DTYPE_MAP.get(TORCH_DTYPE, torch.bfloat16)

    from config import FIGURES_DIR

    if args.plot_only:
        plot_k_sensitivity(model_keys, FIGURES_DIR / "fig_k_sensitivity.pdf")
        plot_norm_blowup(model_keys, FIGURES_DIR / "fig_k_norm_blowup.pdf")
        agg = build_aggregate_sensitivity(model_keys)
        agg.to_csv(OUT_DIR / "aggregate_sensitivity.csv", index=False)
        return

    for mk in model_keys:
        mcfg = MODEL_REGISTRY[mk]
        log.info("=== Loading model: %s ===", mcfg.label)
        tokenizer = AutoTokenizer.from_pretrained(mcfg.hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mcfg.hf_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        run_k_sensitivity(mk, model, tokenizer, device, overwrite=args.overwrite)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=== Generating figures and aggregate summary ===")
    agg = build_aggregate_sensitivity(model_keys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_DIR / "aggregate_sensitivity.csv", index=False)
    plot_k_sensitivity(model_keys, FIGURES_DIR / "fig_k_sensitivity.pdf")
    plot_norm_blowup(model_keys, FIGURES_DIR / "fig_k_norm_blowup.pdf")

    # Report κ=1.0 rank
    for mk in model_keys:
        sub = agg[(agg["model"] == mk) & (agg["is_calibrated"] == True)]
        if len(sub) > 0:
            rank = int(sub["rank_by_fidelity"].iloc[0])
            fid = float(sub["mean_fidelity"].iloc[0])
            total = len(agg[agg["model"] == mk])
            log.info("  %s: κ=1.0 ranks #%d of %d (fidelity=%.4f)", mk, rank, total, fid)

    log.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
