"""
E5_permutation_invariance_v2.py — Permutation invariance & orbit-averaged probe.

Addresses:
  M3: 20 permutation seeds (up from 5), bootstrap CIs on alignment ratios
  MI-F3: Orbit-averaged probe — average behavioral directions across N_orbit
         permuted versions; test if orbit-averaged direction is more invariant
         than any single-permutation direction.

Protocol:
  For each model × behavior:
    1. Extract behavioral PCA direction from the canonical (unpermuted) model.
    2. For each of N_perm=20 seeds:
       a. Randomly permute MLP neurons (fraction p=0.5) in ALL layers simultaneously.
       b. Re-extract the behavioral direction from the permuted model.
       c. Measure |cos(d_canonical, d_permuted)| — "permutation resilience".
    3. Build orbit-averaged direction:
       a. Map each permuted direction back to canonical neuron ordering.
       b. Average all N_orbit=20 directions → d_orbit.
       c. Measure |cos(d_canonical, d_orbit)|.
    4. Bootstrap CI on mean permutation resilience (B=50 subsamples of size 16).

Output columns: layer, seed, cosine_sim, orbit_cosine_sim, is_orbit
Outputs:
  results/permutation_invariance_v2/{model}/{behavior}/permutation_resilience.csv
  results/permutation_invariance_v2/{model}/{behavior}/orbit_summary.csv
  results/permutation_invariance_v2/aggregate_permutation_summary.csv
  figures/fig_permutation_v2.pdf
"""

from __future__ import annotations

import argparse
import gc
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
    ALL_BEHAVIOR_KEYS,
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

OUT_DIR = RESULTS_DIR / "permutation_invariance_v2"
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_contrastive_pairs(behavior_key: str) -> List[Tuple[str, str]]:
    """Load (positive, negative) prompt pairs from behavior data file."""
    bcfg = BEHAVIOR_REGISTRY[behavior_key]
    path = DATA_DIR / "behaviors" / bcfg.data_file
    if not path.exists():
        raise FileNotFoundError(f"Behavior data file not found: {path}")
    import json
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
# Activation extraction
# ---------------------------------------------------------------------------

class ResidualExtractor:
    """
    Hook-based extractor for residual stream activations after each MLP block.
    Handles both pre-norm (llama, qwen, mistral) and dual-norm (gemma2) families.
    """

    def __init__(self, model: nn.Module, architecture_family: str) -> None:
        self.model = model
        self.arch = architecture_family
        self._hooks: list = []
        self.activations: Dict[str, torch.Tensor] = {}

    def _hook_fn(self, name: str):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.activations[name] = out.detach().float().cpu()
        return fn

    def register(self, layer_names: List[str]) -> None:
        layers = self.model.model.layers
        for ln in layer_names:
            idx = int(ln.split("_")[-1])
            hook = layers[idx].register_forward_hook(self._hook_fn(ln))
            self._hooks.append(hook)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()

    def get_layer_names(self, num_layers: int) -> List[str]:
        return [f"layer_{i}" for i in range(num_layers)]


def extract_mean_activation(
    text: str,
    model: nn.Module,
    tokenizer,
    extractor: ResidualExtractor,
    layer_names: List[str],
    device: torch.device,
    max_length: int = 128,
) -> Dict[str, np.ndarray]:
    """Returns mean-pooled residual activations per layer for a single text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)
    extractor.activations.clear()
    with torch.no_grad():
        model(**inputs)
    out: Dict[str, np.ndarray] = {}
    for ln in layer_names:
        act = extractor.activations.get(ln)
        if act is not None:
            out[ln] = act[0].mean(dim=0).numpy()  # (hidden,)
    return out


def compute_diff_matrix(
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    extractor: ResidualExtractor,
    layer_names: List[str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Returns diff_matrix[layer] shape (n_pairs, hidden):
    diff = mean_act(positive) - mean_act(negative).
    """
    n = len(pairs)
    diffs: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}
    for pos_text, neg_text in pairs:
        pos_acts = extract_mean_activation(pos_text, model, tokenizer, extractor, layer_names, device)
        neg_acts = extract_mean_activation(neg_text, model, tokenizer, extractor, layer_names, device)
        for ln in layer_names:
            if ln in pos_acts and ln in neg_acts:
                diffs[ln].append(pos_acts[ln] - neg_acts[ln])
    return {ln: np.stack(diffs[ln], axis=0) for ln in layer_names if diffs[ln]}


# ---------------------------------------------------------------------------
# PCA behavioral direction extraction
# ---------------------------------------------------------------------------

def extract_behavioral_directions(
    diff_matrix: Dict[str, np.ndarray],
    n_components: int = 1,
) -> Dict[str, np.ndarray]:
    """
    PCA over diff matrix rows. Returns PC1 (shape: hidden,) per layer.
    """
    directions: Dict[str, np.ndarray] = {}
    for ln, X in diff_matrix.items():
        if X.shape[0] < 2:
            continue
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        pca.fit(X)
        directions[ln] = pca.components_[0]  # PC1
    return directions


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------

def permute_mlp_neurons(
    model: nn.Module,
    architecture_family: str,
    fraction: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Randomly permutes `fraction` of MLP neurons (gate/up/down projections) in ALL layers.
    Permutation is applied IN-PLACE on model weights.
    Returns a dict mapping layer_idx → permutation index array (for reversal).
    """
    layers = model.model.layers
    perm_map: Dict[str, np.ndarray] = {}

    for i, layer in enumerate(layers):
        mlp = layer.mlp
        # Get intermediate (neuron) dimension
        if architecture_family in ("llama", "qwen2", "mistral"):
            # SwiGLU: gate_proj, up_proj, down_proj
            d_inter = mlp.gate_proj.weight.shape[0]
            n_perm = max(1, int(fraction * d_inter))
            perm_indices = rng.choice(d_inter, size=n_perm, replace=False)
            canonical = np.arange(d_inter)
            shuffled = canonical.copy()
            shuffled[perm_indices] = rng.permutation(perm_indices)
            # Apply permutation: reorder neuron dimension
            with torch.no_grad():
                mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[shuffled]
                mlp.up_proj.weight.data = mlp.up_proj.weight.data[shuffled]
                mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, shuffled]
            perm_map[f"layer_{i}"] = shuffled
        elif architecture_family == "gemma2":
            # Same SwiGLU structure
            d_inter = mlp.gate_proj.weight.shape[0]
            n_perm = max(1, int(fraction * d_inter))
            perm_indices = rng.choice(d_inter, size=n_perm, replace=False)
            canonical = np.arange(d_inter)
            shuffled = canonical.copy()
            shuffled[perm_indices] = rng.permutation(perm_indices)
            with torch.no_grad():
                mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[shuffled]
                mlp.up_proj.weight.data = mlp.up_proj.weight.data[shuffled]
                mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, shuffled]
            perm_map[f"layer_{i}"] = shuffled
    return perm_map


def reverse_permutation(
    model: nn.Module,
    architecture_family: str,
    perm_map: Dict[str, np.ndarray],
) -> None:
    """Reverses the in-place permutation by applying the inverse permutation."""
    layers = model.model.layers
    for i, layer in enumerate(layers):
        ln = f"layer_{i}"
        if ln not in perm_map:
            continue
        shuffled = perm_map[ln]
        inv_perm = np.argsort(shuffled)
        mlp = layer.mlp
        if architecture_family in ("llama", "qwen2", "mistral", "gemma2"):
            with torch.no_grad():
                mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[inv_perm]
                mlp.up_proj.weight.data = mlp.up_proj.weight.data[inv_perm]
                mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, inv_perm]


# ---------------------------------------------------------------------------
# Core experiment: permutation resilience
# ---------------------------------------------------------------------------

def run_permutation_resilience(
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    tokenizer,
    model_cfg,
    device: torch.device,
    n_perm_seeds: int,
    permutation_fraction: float,
    n_orbit_samples: int,
    behavior_key: str,
    out_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Main permutation invariance analysis.

    Returns DataFrame with per-layer permutation resilience scores
    and orbit-averaged direction cosine similarity.
    """
    resilience_path = out_dir / "permutation_resilience.csv"
    orbit_path = out_dir / "orbit_summary.csv"

    if resilience_path.exists() and orbit_path.exists() and not overwrite:
        log.info("Checkpoint found — skipping %s / %s", model_cfg.key, behavior_key)
        return pd.read_csv(resilience_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers = model_cfg.num_layers
    arch = model_cfg.architecture_family

    extractor = ResidualExtractor(model, arch)
    layer_names = extractor.get_layer_names(num_layers)
    extractor.register(layer_names)

    log.info("Extracting canonical behavioral directions...")
    diff_canonical = compute_diff_matrix(pairs, model, tokenizer, extractor, layer_names, device)
    canonical_dirs = extract_behavioral_directions(diff_canonical)
    log.info("  Canonical directions extracted for %d layers.", len(canonical_dirs))

    records: List[dict] = []
    # Accumulate orbit directions: layer → list of direction vectors
    orbit_dirs: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}

    rng = np.random.default_rng(GLOBAL_SEED)

    for seed in tqdm(range(n_perm_seeds), desc=f"Perm seeds [{model_cfg.key}/{behavior_key}]"):
        seed_rng = np.random.default_rng(GLOBAL_SEED + seed + 1)

        # Permute in-place
        perm_map = permute_mlp_neurons(model, arch, permutation_fraction, seed_rng)

        # Extract directions on permuted model
        diff_perm = compute_diff_matrix(pairs, model, tokenizer, extractor, layer_names, device)
        perm_dirs = extract_behavioral_directions(diff_perm)

        for ln in layer_names:
            if ln not in canonical_dirs or ln not in perm_dirs:
                continue
            canon = canonical_dirs[ln]
            perm_d = perm_dirs[ln]
            cos_sim = float(np.abs(np.dot(canon, perm_d) / (np.linalg.norm(canon) * np.linalg.norm(perm_d) + 1e-10)))
            records.append({
                "layer": ln,
                "seed": seed,
                "cosine_sim": cos_sim,
                "is_orbit": False,
            })
            if seed < n_orbit_samples:
                orbit_dirs[ln].append(perm_dirs[ln])

        # Restore weights
        reverse_permutation(model, arch, perm_map)

        if seed % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    extractor.remove()

    # Build orbit-averaged direction and compute cosine sim with canonical
    orbit_records: List[dict] = []
    for ln in layer_names:
        if ln not in canonical_dirs or not orbit_dirs[ln]:
            continue
        orbit_stack = np.stack(orbit_dirs[ln], axis=0)  # (n_orbit, hidden)
        orbit_mean = orbit_stack.mean(axis=0)
        orbit_mean_norm = orbit_mean / (np.linalg.norm(orbit_mean) + 1e-10)
        canon = canonical_dirs[ln]
        canon_norm = canon / (np.linalg.norm(canon) + 1e-10)
        orbit_cos = float(np.abs(np.dot(canon_norm, orbit_mean_norm)))
        orbit_records.append({
            "layer": ln,
            "orbit_cosine_sim": orbit_cos,
            "n_orbit_samples": len(orbit_dirs[ln]),
        })

    resilience_df = pd.DataFrame(records)
    orbit_df = pd.DataFrame(orbit_records)
    resilience_df.to_csv(resilience_path, index=False)
    orbit_df.to_csv(orbit_path, index=False)
    log.info("Saved: %s", resilience_path)
    log.info("Saved: %s", orbit_path)

    return resilience_df


# ---------------------------------------------------------------------------
# Bootstrap CI on mean permutation resilience
# ---------------------------------------------------------------------------

def bootstrap_permutation_resilience(
    resilience_df: pd.DataFrame,
    B: int = 50,
    subsample_size: int = 16,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Percentile bootstrap CI on mean cosine_sim across layers.
    Returns DataFrame with columns: mean, ci_lo, ci_hi, std.
    """
    rng = np.random.default_rng(seed)
    sims = resilience_df.groupby("layer")["cosine_sim"].mean().values
    if len(sims) == 0:
        return pd.DataFrame()

    bootstrap_means: List[float] = []
    for _ in range(B):
        indices = rng.choice(len(sims), size=min(subsample_size, len(sims)), replace=True)
        bootstrap_means.append(float(sims[indices].mean()))

    bootstrap_means_arr = np.array(bootstrap_means)
    return pd.DataFrame([{
        "mean_cosine_sim": float(sims.mean()),
        "ci_lo": float(np.percentile(bootstrap_means_arr, 2.5)),
        "ci_hi": float(np.percentile(bootstrap_means_arr, 97.5)),
        "std": float(bootstrap_means_arr.std()),
    }])


# ---------------------------------------------------------------------------
# Aggregate + LaTeX
# ---------------------------------------------------------------------------

def build_aggregate_summary(model_keys: List[str], behavior_keys: List[str]) -> pd.DataFrame:
    """Reads all saved CSVs and builds a comprehensive aggregate table."""
    records: List[dict] = []
    for mk in model_keys:
        for bk in behavior_keys:
            resil_path = OUT_DIR / mk / bk / "permutation_resilience.csv"
            orbit_path = OUT_DIR / mk / bk / "orbit_summary.csv"
            if not resil_path.exists():
                continue
            df_resil = pd.read_csv(resil_path)
            mean_sim = df_resil["cosine_sim"].mean()
            std_sim = df_resil["cosine_sim"].std()
            boot_df = bootstrap_permutation_resilience(df_resil)

            orbit_mean = float("nan")
            if orbit_path.exists():
                orbit_df = pd.read_csv(orbit_path)
                orbit_mean = orbit_df["orbit_cosine_sim"].mean()

            records.append({
                "model": mk,
                "behavior": bk,
                "mean_cosine_sim": mean_sim,
                "std_cosine_sim": std_sim,
                "ci_lo": boot_df["ci_lo"].iloc[0] if len(boot_df) else float("nan"),
                "ci_hi": boot_df["ci_hi"].iloc[0] if len(boot_df) else float("nan"),
                "orbit_cosine_sim": orbit_mean,
                "orbit_improvement": orbit_mean - mean_sim if not np.isnan(orbit_mean) else float("nan"),
            })
    return pd.DataFrame(records)


def build_latex_permutation_table(agg_df: pd.DataFrame) -> str:
    """LaTeX table: Model | Behavior | Mean ± CI | Orbit Sim"""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{",
        r"    \textbf{Permutation resilience of behavioral directions.}",
        r"    Mean $|\cos(\hat{d}_{\text{canonical}}, \hat{d}_{\text{permuted}})|$",
        r"    across 20 random neuron permutations ($p=0.5$ of MLP neurons per layer).",
        r"    Orbit column: cosine similarity of the orbit-averaged direction",
        r"    to the canonical direction. 95\% CI from percentile bootstrap ($B=50$).",
        r"  }",
        r"  \label{tab:permutation_resilience}",
        r"  \vskip 0.05in",
        r"  \resizebox{\columnwidth}{!}{%",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Behavior}",
        r"    & \textbf{Mean Resilience} & \textbf{95\% CI} & \textbf{Orbit Sim} \\",
        r"    \midrule",
    ]
    prev_model = None
    for _, row in agg_df.sort_values(["model", "behavior"]).iterrows():
        model_cell = row["model"] if row["model"] != prev_model else ""
        prev_model = row["model"]
        mean_s = f"${row['mean_cosine_sim']:.3f}$"
        ci_s = f"$[{row['ci_lo']:.3f},\\,{row['ci_hi']:.3f}]$"
        orbit_s = f"${row['orbit_cosine_sim']:.3f}$" if not np.isnan(row["orbit_cosine_sim"]) else "---"
        lines.append(f"    {model_cell} & {row['behavior']} & {mean_s} & {ci_s} & {orbit_s} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_permutation_resilience(agg_df: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: x=behavior, bars=models, height=mean cosine sim,
    error bars=95% CI. Horizontal dashed line at orbit-mean cosine sim.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available — skipping figure generation.")
        return

    behaviors = agg_df["behavior"].unique().tolist()
    models = agg_df["model"].unique().tolist()
    n_behaviors = len(behaviors)
    n_models = len(models)
    bar_width = 0.15
    x = np.arange(n_behaviors)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, mk in enumerate(models):
        sub = agg_df[agg_df["model"] == mk].set_index("behavior")
        vals = [sub.loc[b, "mean_cosine_sim"] if b in sub.index else 0.0 for b in behaviors]
        lo_errs = [vals[bi] - sub.loc[b, "ci_lo"] if b in sub.index else 0.0 for bi, b in enumerate(behaviors)]
        hi_errs = [sub.loc[b, "ci_hi"] - vals[bi] if b in sub.index else 0.0 for bi, b in enumerate(behaviors)]
        ax.bar(
            x + mi * bar_width, vals, bar_width,
            label=MODEL_REGISTRY[mk].label if mk in MODEL_REGISTRY else mk,
            color=colors[mi % len(colors)],
            yerr=[lo_errs, hi_errs],
            capsize=3,
            error_kw={"linewidth": 1.0},
        )

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([b.replace("_", "\n") for b in behaviors], fontsize=9)
    ax.set_ylabel(r"$|\cos(\hat{d}_\mathrm{can}, \hat{d}_\mathrm{perm})|$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (0.5)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Permutation Resilience of Behavioral Directions (20 seeds, $p=0.5$)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5: Permutation invariance v2")
    p.add_argument("--model", choices=list(ALL_MODEL_KEYS) + ["all"], default="all")
    p.add_argument("--behavior", choices=list(ALL_BEHAVIOR_KEYS) + ["all"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = list(ALL_MODEL_KEYS) if args.model == "all" else [args.model]
    behavior_keys = list(ALL_BEHAVIOR_KEYS) if args.behavior == "all" else [args.behavior]
    device = torch.device(args.device)
    dtype = DTYPE_MAP.get(TORCH_DTYPE, torch.bfloat16)

    if args.aggregate_only:
        log.info("=== Aggregate-only mode ===")
        agg = build_aggregate_summary(model_keys, behavior_keys)
        agg.to_csv(OUT_DIR / "aggregate_permutation_summary.csv", index=False)
        latex = build_latex_permutation_table(agg)
        (OUT_DIR / "latex_permutation_table.tex").write_text(latex, encoding="utf-8")
        from config import FIGURES_DIR
        plot_permutation_resilience(agg, FIGURES_DIR / "fig_permutation_v2.pdf")
        log.info("Done. Outputs in %s", OUT_DIR)
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
        log.info("  Model loaded on device: %s", next(model.parameters()).device)

        for bk in behavior_keys:
            log.info("--- Behavior: %s ---", bk)
            try:
                pairs = load_contrastive_pairs(bk)
                log.info("  Loaded %d contrastive pairs", len(pairs))
            except FileNotFoundError as e:
                log.warning("  Skipping: %s", e)
                continue

            bdir = OUT_DIR / mk / bk
            run_permutation_resilience(
                pairs=pairs,
                model=model,
                tokenizer=tokenizer,
                model_cfg=mcfg,
                device=device,
                n_perm_seeds=EXPERIMENT_CFG.n_permutation_seeds,
                permutation_fraction=EXPERIMENT_CFG.permutation_fraction,
                n_orbit_samples=EXPERIMENT_CFG.n_orbit_samples,
                behavior_key=bk,
                out_dir=bdir,
                overwrite=args.overwrite,
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=== Building aggregate summary ===")
    agg = build_aggregate_summary(model_keys, behavior_keys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_DIR / "aggregate_permutation_summary.csv", index=False)
    latex = build_latex_permutation_table(agg)
    (OUT_DIR / "latex_permutation_table.tex").write_text(latex, encoding="utf-8")
    from config import FIGURES_DIR
    plot_permutation_resilience(agg, FIGURES_DIR / "fig_permutation_v2.pdf")
    log.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
