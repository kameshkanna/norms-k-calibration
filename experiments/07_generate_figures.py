"""
Generate all paper figures from experiment results.

Produces 6 figures saved to figures/:
  fig1_k_vs_spectral.pdf      — K vs W_up spectral norm scatter (4-panel)
  fig2_alignment_heatmap.pdf  — Weight-space alignment ratio heatmap
  fig3_norm_profiles.pdf      — Per-layer activation norm profiles
  fig4_efficacy.pdf           — Steering method comparison bar chart
  fig5_permutation.pdf        — Permutation invariance cosine-sim distribution
  fig6_raw_control.pdf        — Behavioral vs. generic direction W_down alignment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
FIGURES.mkdir(exist_ok=True)

MODEL_KEYS = ["llama", "qwen", "mistral", "gemma"]
MODEL_LABELS = {
    "llama": "Llama 3.1 8B",
    "qwen": "Qwen 2.5 7B",
    "mistral": "Mistral 7B",
    "gemma": "Gemma 2 9B",
}
BEHAVIORS = [
    "refusal_calibration",
    "formality",
    "verbosity_control",
    "uncertainty_expression",
    "sycophancy_suppression",
]
BEHAVIOR_LABELS = {
    "refusal_calibration": "Refusal",
    "formality": "Formality",
    "verbosity_control": "Verbosity",
    "uncertainty_expression": "Uncertainty",
    "sycophancy_suppression": "Sycophancy",
}
MODEL_DIR_MAP = {
    "llama": "meta-llama__Llama-3.1-8B-Instruct",
    "qwen": "Qwen__Qwen2.5-7B-Instruct",
    "mistral": "mistralai__Mistral-7B-Instruct-v0.3",
    "gemma": "google__gemma-2-9b-it",
}

# Publication-quality style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("colorblind", 4)


# ---------------------------------------------------------------------------
# Figure 1: K vs spectral norm scatter (4-panel)
# ---------------------------------------------------------------------------

def fig1_k_vs_spectral() -> None:
    """Scatter K_ℓ vs σ₁(W_up) per model with Pearson r annotated."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes = axes.flatten()

    for ax, (model, color) in zip(axes, zip(MODEL_KEYS, PALETTE)):
        csv = RESULTS / "k_calibration" / f"{model}_k_vs_spectral.csv"
        corr = RESULTS / "k_calibration" / f"{model}_correlation.json"
        df = pd.read_csv(csv)
        with open(corr) as f:
            meta = json.load(f)

        r = meta["pearson_r_up_proj"]
        p = meta["pearson_p_up_proj"]
        n = meta["n_layers_used_up_proj"]

        # Normalize layer depth to [0, 1] for color encoding
        depth = df["layer_idx"] / df["layer_idx"].max()

        sc = ax.scatter(
            df["spectral_norm_up_proj"],
            df["k_value"],
            c=depth,
            cmap="viridis",
            s=50,
            alpha=0.85,
            edgecolors="none",
        )

        # Regression line
        slope, intercept, *_ = stats.linregress(
            df["spectral_norm_up_proj"], df["k_value"]
        )
        x_range = np.linspace(df["spectral_norm_up_proj"].min(), df["spectral_norm_up_proj"].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, color="firebrick", lw=1.5, ls="--")

        p_str = f"{p:.0e}" if p < 0.001 else f"{p:.3f}"
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\sigma_1(W_\mathrm{up})$", fontsize=10)
        ax.set_ylabel(r"$K_\ell$", fontsize=10)
        ax.annotate(
            f"$r = {r:.3f}$\n$p = {p_str}$\n$n = {n}$",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Layer depth", fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Early", "Mid", "Late"])

    fig.suptitle(
        r"$K_\ell = \bar\mu_\ell / \sqrt{d}$ vs. $\sigma_1(W_\mathrm{up})$ per layer",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    out = FIGURES / "fig1_k_vs_spectral.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 2: Weight-space alignment heatmap
# ---------------------------------------------------------------------------

def fig2_alignment_heatmap() -> None:
    """Heatmap of mean alignment ratios (PCA vs SVD) across models x behaviors."""
    data: list[dict] = []

    for model in MODEL_KEYS:
        model_dir = MODEL_DIR_MAP[model]
        for behavior in BEHAVIORS:
            csv_path = RESULTS / "weight_alignment" / model_dir / behavior / "alignment_per_layer.csv"
            if not csv_path.exists():
                logger.warning("Missing: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            mean_ratio = df["alignment_ratio"].mean()
            data.append({"Model": MODEL_LABELS[model], "Behavior": BEHAVIOR_LABELS[behavior], "Alignment ratio": mean_ratio})

    if not data:
        logger.error("No alignment data found — skipping fig2")
        return

    df_heat = pd.DataFrame(data).pivot(index="Model", columns="Behavior", values="Alignment ratio")
    # Reorder to match paper model order
    df_heat = df_heat.reindex([MODEL_LABELS[m] for m in MODEL_KEYS])

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Alignment ratio (× random baseline)"},
        vmin=1.0,
    )
    ax.set_title(
        "PCA direction alignment with $W_\\mathrm{down}$ singular vectors\n(ratio above random baseline)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    out = FIGURES / "fig2_alignment_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 3: Per-layer activation norm profiles
# ---------------------------------------------------------------------------

def fig3_norm_profiles() -> None:
    """Line plot of mean L2 norm per layer for all 4 models."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for model, color in zip(MODEL_KEYS, PALETTE):
        csv = RESULTS / "norm_profiles" / f"{model}.csv"
        if not csv.exists():
            logger.warning("Missing norm profile: %s", csv)
            continue
        df = pd.read_csv(csv)
        ax.plot(df["layer_idx"], df["mean_norm"], label=MODEL_LABELS[model], color=color, lw=2)
        ax.fill_between(
            df["layer_idx"],
            df["mean_norm"] - df["std_norm"],
            df["mean_norm"] + df["std_norm"],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(r"Mean L2 norm $\bar\mu_\ell$", fontsize=11)
    ax.set_title("Per-layer residual-stream activation norms", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(which="minor", alpha=0.2)

    out = FIGURES / "fig3_norm_profiles.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 4: Efficacy comparison bar chart
# ---------------------------------------------------------------------------

def fig4_efficacy() -> None:
    """Grouped bar chart: accuracy per method, grouped by behavior (aggregated over models)."""
    records: list[dict] = []
    method_order = ["none", "raw_addition", "pca_uncalibrated", "pca_k_calibrated"]
    method_labels = {
        "none": "No steering",
        "raw_addition": "Raw addition",
        "pca_uncalibrated": "PCA (uncal.)",
        "pca_k_calibrated": "PCA + K-cal.",
    }

    for model in MODEL_KEYS:
        model_dir = MODEL_DIR_MAP[model]
        for behavior in BEHAVIORS:
            csv_path = RESULTS / "efficacy" / model_dir / behavior / "comparison.csv"
            if not csv_path.exists():
                logger.warning("Missing efficacy: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                records.append({
                    "model": model,
                    "behavior": BEHAVIOR_LABELS[behavior],
                    "method": row["method"],
                    "accuracy": row["accuracy"],
                })

    if not records:
        logger.error("No efficacy data found — skipping fig4")
        return

    df_all = pd.DataFrame(records)
    # Aggregate over models: mean accuracy per (behavior, method)
    df_agg = df_all.groupby(["behavior", "method"])["accuracy"].mean().reset_index()

    # Build pivot for grouped bar
    df_pivot = df_agg.pivot(index="behavior", columns="method", values="accuracy")
    df_pivot = df_pivot[[m for m in method_order if m in df_pivot.columns]]
    df_pivot.columns = [method_labels[c] for c in df_pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    bar_colors = sns.color_palette("colorblind", len(df_pivot.columns))
    df_pivot.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white", width=0.75)

    ax.set_xlabel("Behavioral axis", fontsize=11)
    ax.set_ylabel("Accuracy (fraction correct)", fontsize=11)
    ax.set_title("Steering efficacy by method and behavior\n(mean over 4 models, 9 test pairs each)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chance (0.5)")
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    out = FIGURES / "fig4_efficacy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 5: Permutation invariance distribution
# ---------------------------------------------------------------------------

def fig5_permutation() -> None:
    """Box + strip plot of subspace cosine similarities per model under permutation."""
    records: list[dict] = []

    for model in MODEL_KEYS:
        for behavior in BEHAVIORS:
            csv_path = RESULTS / "permutation_invariance" / model / behavior / "invariance_scores.csv"
            if not csv_path.exists():
                logger.warning("Missing permutation: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            for val in df["subspace_cosine_sim"]:
                records.append({"Model": MODEL_LABELS[model], "Cosine similarity": float(val)})

    if not records:
        logger.error("No permutation invariance data — skipping fig5")
        return

    df_perm = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sns.boxplot(
        data=df_perm,
        x="Model",
        y="Cosine similarity",
        hue="Model",
        palette=PALETTE,
        legend=False,
        ax=ax,
        width=0.45,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        linewidth=1.2,
    )
    ax.axhline(0.85, color="firebrick", ls="--", lw=1.5, label="Invariance threshold (0.85)")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title(
        "Subspace cosine similarity after 50% neuron permutation\n(lower = more sensitive to permutation)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Subspace cosine similarity", fontsize=11)
    ax.legend(fontsize=9)

    out = FIGURES / "fig5_permutation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 6: Raw activation control — behavioral vs. generic W_down alignment
# ---------------------------------------------------------------------------

def fig6_raw_control() -> None:
    """Grouped bar chart comparing W_down alignment ratios across five direction types.

    Source: results/raw_activation_control/aggregate_summary.csv
    Columns: model_key, behavioral_pc1_ratio, contrastive_mean_ratio,
             raw_pc1_ratio, raw_mean_dir_ratio, random_baseline,
             behavioral_advantage_over_raw_pc1
    """
    csv_path = RESULTS / "raw_activation_control" / "aggregate_summary.csv"
    if not csv_path.exists():
        logger.error("Missing aggregate summary: %s — skipping fig6", csv_path)
        return

    df = pd.read_csv(csv_path)
    df["model_label"] = df["model_key"].map(MODEL_LABELS)
    df = df.set_index("model_key").reindex(MODEL_KEYS).reset_index()

    # Direction types to plot as grouped bars (raw_mean excluded from main bars —
    # shown as a separate hatched overlay to mark it as a structural artifact)
    bar_directions = [
        ("behavioral_pc1_ratio",   "Behavioral PC1",    "#2166ac", None),
        ("contrastive_mean_ratio",  "Contrastive mean",  "#4dac26", None),
        ("raw_pc1_ratio",           "Raw PC1",           "#d01c8b", None),
        ("raw_mean_dir_ratio",      "Raw mean (artifact)", "#b2b2b2", "////"),
    ]

    n_models = len(MODEL_KEYS)
    n_bars = len(bar_directions)
    bar_width = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    for i, (col, label, color, hatch) in enumerate(bar_directions):
        offsets = x + (i - (n_bars - 1) / 2) * bar_width
        vals = df[col].to_numpy()
        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="white" if hatch is None else "#666666",
            linewidth=0.8,
            alpha=0.88,
        )
        # Annotate behavioral advantage above behavioral_pc1 bars
        if col == "behavioral_pc1_ratio":
            for bar, (_, row) in zip(bars, df.iterrows()):
                adv = row["behavioral_advantage_over_raw_pc1"]
                sign = "+" if adv >= 0 else ""
                color_adv = "#1a6320" if adv >= 0 else "#991111"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{sign}{adv:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=color_adv,
                    fontweight="bold",
                )

    # Random baseline as a horizontal reference line per model group
    for xi, (_, row) in zip(x, df.iterrows()):
        span = bar_width * n_bars / 2 + bar_width * 0.3
        ax.hlines(
            row["random_baseline"],
            xi - span,
            xi + span,
            colors="#999999",
            linestyles=":",
            linewidths=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_KEYS], fontsize=10)
    ax.set_ylabel(r"$W_\mathrm{down}$ alignment ratio (× random baseline)", fontsize=10)
    ax.set_title(
        r"Contrastive extraction yields $W_\mathrm{down}$-aligned directions"
        "\nabove the generic activation baseline (3 of 4 architectures)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axhline(1.0, color="#444444", ls="--", lw=1.0, alpha=0.6, label="Random baseline (1.0×)")
    ax.set_ylim(0, df["raw_mean_dir_ratio"].max() * 1.18)

    # Gemma 2 annotation — explain the exception
    gemma_idx = MODEL_KEYS.index("gemma")
    ax.annotate(
        "Post-norm\ndecoupling",
        xy=(gemma_idx, df.loc[df["model_key"] == "gemma", "behavioral_pc1_ratio"].values[0] + 0.3),
        xytext=(gemma_idx + 0.55, df.loc[df["model_key"] == "gemma", "behavioral_pc1_ratio"].values[0] + 2.5),
        fontsize=8,
        color="#7b3294",
        arrowprops=dict(arrowstyle="->", color="#7b3294", lw=1.0),
        ha="center",
    )

    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out = FIGURES / "fig6_raw_control.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Generating paper figures → %s", FIGURES)
    fig1_k_vs_spectral()
    fig2_alignment_heatmap()
    fig3_norm_profiles()
    fig4_efficacy()
    fig5_permutation()
    fig6_raw_control()
    logger.info("Done. Figures written to %s", FIGURES)


if __name__ == "__main__":
    main()
