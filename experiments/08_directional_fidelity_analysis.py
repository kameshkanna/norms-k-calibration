"""
Experiment 08: Directional Fidelity Analysis (Appendix B).

Aggregates mean_cosine_shift from all efficacy comparison CSVs across
4 models × 5 behaviors × 4 methods. Produces:
  - results/directional_fidelity/aggregate_table.csv
  - results/directional_fidelity/latex_table.tex
  - figures/fig_appendix_b_directional_fidelity.pdf

Key comparison: pca_k_calibrated vs pca_uncalibrated isolates the effect
of K-calibration alone (same behavioral direction, different magnitude).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EFFICACY_DIR = ROOT / "results" / "efficacy"
OUT_DIR = ROOT / "results" / "directional_fidelity"
FIG_DIR = ROOT / "figures"

MODEL_DIRS: dict[str, str] = {
    "meta-llama__Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "google__gemma-2-9b-it": "Gemma 2 9B",
    "Qwen__Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "mistralai__Mistral-7B-Instruct-v0.3": "Mistral 7B",
}

BEHAVIORS: list[str] = [
    "formality",
    "refusal_calibration",
    "sycophancy_suppression",
    "uncertainty_expression",
    "verbosity_control",
]

BEHAVIOR_LABELS: dict[str, str] = {
    "formality": "Formality",
    "refusal_calibration": "Refusal",
    "sycophancy_suppression": "Sycophancy",
    "uncertainty_expression": "Uncertainty",
    "verbosity_control": "Verbosity",
}

METHOD_ORDER: list[str] = [
    "none",
    "raw_addition",
    "pca_uncalibrated",
    "pca_k_calibrated",
]

METHOD_LABELS: dict[str, str] = {
    "none": "Baseline (no steering)",
    "raw_addition": "Raw addition (K=1)",
    "pca_uncalibrated": "Behavioral PC1 (K=1)",
    "pca_k_calibrated": r"Behavioral PC1 (K=$\bar{\mu}/\sqrt{d}$)",
}

METHOD_COLORS: dict[str, str] = {
    "none": "#aaaaaa",
    "raw_addition": "#e07b39",
    "pca_uncalibrated": "#4878cf",
    "pca_k_calibrated": "#6acc65",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    """Load all comparison CSVs into a single tidy DataFrame."""
    records: list[dict] = []
    for model_dir, model_label in MODEL_DIRS.items():
        for behavior in BEHAVIORS:
            csv_path = EFFICACY_DIR / model_dir / behavior / "comparison.csv"
            if not csv_path.exists():
                log.warning("Missing: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                records.append(
                    {
                        "model": model_label,
                        "behavior": behavior,
                        "method": row["method"],
                        "mean_cosine_shift": float(row["mean_cosine_shift"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )
    result = pd.DataFrame(records)
    log.info("Loaded %d rows from %d model×behavior combinations.", len(result), len(MODEL_DIRS) * len(BEHAVIORS))
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model per-method: mean and std of mean_cosine_shift across behaviors.
    Also compute gain of pca_k_calibrated over pca_uncalibrated.
    """
    agg = (
        df.groupby(["model", "method"])["mean_cosine_shift"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mcs_mean", "std": "mcs_std", "count": "n"})
    )

    # Standard error
    agg["mcs_sem"] = agg["mcs_std"] / np.sqrt(agg["n"])

    # 95% CI halfwidth (t-distribution, df=n-1=4)
    from scipy import stats as scipy_stats
    t_crit = scipy_stats.t.ppf(0.975, df=4)
    agg["ci95"] = agg["mcs_sem"] * t_crit

    return agg


def compute_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model, compute absolute and relative gain of
    pca_k_calibrated over pca_uncalibrated (averaged over behaviors).
    """
    pivot = (
        df.groupby(["model", "method"])["mean_cosine_shift"]
        .mean()
        .unstack("method")
        .reset_index()
    )

    models_order = list(MODEL_DIRS.values())
    pivot["model"] = pd.Categorical(pivot["model"], categories=models_order, ordered=True)
    pivot = pivot.sort_values("model").reset_index(drop=True)

    pivot["abs_gain"] = pivot["pca_k_calibrated"] - pivot["pca_uncalibrated"]
    pivot["rel_gain_pct"] = (pivot["abs_gain"] / pivot["pca_uncalibrated"].abs()) * 100
    pivot["vs_raw_abs"] = pivot["pca_k_calibrated"] - pivot["raw_addition"]

    return pivot


# ---------------------------------------------------------------------------
# Wilson CI for accuracy (Appendix note)
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - margin, center + margin


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def build_latex_table(agg: pd.DataFrame, gain: pd.DataFrame) -> str:
    """
    Build a LaTeX booktabs table with one row per model, one column group per method.
    Reports mean_cosine_shift ± CI95 for the two key methods plus K-cal gain.
    """
    models_order = list(MODEL_DIRS.values())

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{")
    lines.append(r"    \textbf{Directional fidelity (mean cosine shift, $\uparrow$ better)} across")
    lines.append(r"    4 models and 5 behavioral axes. Values are mean $\pm$ 95\% CI over")
    lines.append(r"    behaviors ($n{=}5$). \textit{Gain} is the absolute improvement of")
    lines.append(r"    K-calibrated over uncalibrated; $\dagger$ marks cases where post-norm")
    lines.append(r"    decoupling reduces absolute shift (see Remark~3).")
    lines.append(r"  }")
    lines.append(r"  \label{tab:directional_fidelity}")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Model}")
    lines.append(r"    & \textbf{Baseline}")
    lines.append(r"    & \textbf{Raw (K=1)}")
    lines.append(r"    & \textbf{PC1 (K=1)}")
    lines.append(r"    & \textbf{PC1 (K-cal.)} \\")
    lines.append(r"    \midrule")

    for model in models_order:
        model_agg = agg[agg["model"] == model].set_index("method")
        gain_row = gain[gain["model"] == model].iloc[0]

        def fmt(method: str) -> str:
            if method not in model_agg.index:
                return r"---"
            row = model_agg.loc[method]
            val = row["mcs_mean"]
            ci = row["ci95"]
            return rf"${val:.4f}_{{\pm{ci:.4f}}}$"

        # Mark Gemma K-calibrated with dagger (post-norm decoupling)
        kcal_str = fmt("pca_k_calibrated")
        if model == "Gemma 2 9B":
            kcal_str = kcal_str.rstrip("$") + r"^{\dagger}$"

        gain_val = gain_row["abs_gain"]
        gain_sign = "+" if gain_val >= 0 else ""
        gain_str = rf"$\mathbf{{{gain_sign}{gain_val:.4f}}}$"

        line = (
            rf"    {model} & {fmt('none')} & {fmt('raw_addition')} "
            rf"& {fmt('pca_uncalibrated')} & {kcal_str} \\"
        )
        lines.append(line)

    lines.append(r"    \midrule")

    # Average row
    avg_agg = agg.groupby("method")["mcs_mean"].mean()
    avg_ci = agg.groupby("method")["ci95"].mean()

    def fmt_avg(method: str) -> str:
        val = avg_agg.get(method, float("nan"))
        ci = avg_ci.get(method, float("nan"))
        return rf"${val:.4f}_{{\pm{ci:.4f}}}$"

    lines.append(
        rf"    \textit{{Average}} & {fmt_avg('none')} & {fmt_avg('raw_addition')} "
        rf"& {fmt_avg('pca_uncalibrated')} & {fmt_avg('pca_k_calibrated')} \\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_directional_fidelity(df: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: x=model, groups=method, y=mean_cosine_shift.
    Error bars = 95% CI over behaviors.
    """
    models_order = list(MODEL_DIRS.values())
    methods = ["none", "raw_addition", "pca_uncalibrated", "pca_k_calibrated"]

    agg = (
        df.groupby(["model", "method"])["mean_cosine_shift"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    from scipy import stats as scipy_stats
    t_crit = scipy_stats.t.ppf(0.975, df=4)
    agg["ci95"] = agg["sem"] * t_crit

    n_models = len(models_order)
    n_methods = len(methods)
    bar_width = 0.18
    group_gap = 0.05
    group_width = n_methods * bar_width + group_gap

    x_centers = np.arange(n_models) * group_width

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, method in enumerate(methods):
        offsets = x_centers + (i - n_methods / 2 + 0.5) * bar_width
        vals, errs = [], []
        for model in models_order:
            row = agg[(agg["model"] == model) & (agg["method"] == method)]
            vals.append(row["mean"].values[0] if len(row) else 0.0)
            errs.append(row["ci95"].values[0] if len(row) else 0.0)

        hatch = "//" if method == "none" else ""
        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            color=METHOD_COLORS[method],
            hatch=hatch,
            label=METHOD_LABELS[method],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            yerr=errs,
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "#333333"},
            zorder=3,
        )

        # Annotate K-calibrated bars with gain over uncalibrated
        if method == "pca_k_calibrated":
            unc_vals = []
            for model in models_order:
                row_u = agg[(agg["model"] == model) & (agg["method"] == "pca_uncalibrated")]
                unc_vals.append(row_u["mean"].values[0] if len(row_u) else 0.0)
            for bar, val, unc, err in zip(bars, vals, unc_vals, errs):
                gain = val - unc
                sign = "+" if gain >= 0 else ""
                color = "#2d6a2d" if gain >= 0 else "#8b0000"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(val, 0) + err + 0.002,
                    f"{sign}{gain:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

    # Gemma post-norm annotation
    gemma_idx = models_order.index("Gemma 2 9B")
    ax.annotate(
        "Post-norm\ndecoupling\n(Remark 3)",
        xy=(x_centers[gemma_idx], 0.004),
        xytext=(x_centers[gemma_idx] + 0.15, 0.025),
        fontsize=7,
        color="#7b2d8b",
        arrowprops=dict(arrowstyle="->", color="#7b2d8b", lw=0.8),
        ha="left",
    )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(models_order, fontsize=10)
    ax.set_ylabel("Mean cosine shift (directional fidelity $\\uparrow$)", fontsize=10)
    ax.set_xlabel("")
    ax.set_title(
        "K-calibration improves directional fidelity across 3 of 4 architectures\n"
        r"$\it{Kicker: K-calibrated\ perturbation\ stays\ in\ the\ behavioral\ subspace.}$",
        fontsize=10,
        pad=10,
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
        ncol=2,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Per-behavior breakdown heatmap
# ---------------------------------------------------------------------------

def plot_behavior_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """
    Heatmap: rows = model × behavior, cols = method.
    Cell = mean_cosine_shift. K-calibrated gain highlighted.
    """
    methods = ["raw_addition", "pca_uncalibrated", "pca_k_calibrated"]
    models_order = list(MODEL_DIRS.values())

    rows_index: list[str] = []
    data_rows: list[list[float]] = []

    for model in models_order:
        for behavior in BEHAVIORS:
            key = f"{model} / {BEHAVIOR_LABELS[behavior]}"
            rows_index.append(key)
            row_vals = []
            for method in methods:
                sub = df[(df["model"] == model) & (df["behavior"] == behavior) & (df["method"] == method)]
                row_vals.append(sub["mean_cosine_shift"].values[0] if len(sub) else float("nan"))
            data_rows.append(row_vals)

    mat = np.array(data_rows)
    col_labels = [
        "Raw (K=1)",
        "PC1 (K=1)",
        r"PC1 (K-cal.)",
    ]

    fig, ax = plt.subplots(figsize=(6, 8))
    vmax = np.nanpercentile(np.abs(mat), 95)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(rows_index)))
    ax.set_yticklabels(rows_index, fontsize=7.5)

    # Draw model separator lines
    for sep in [5, 10, 15]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6.5, color="black")

    ax.set_title(
        "Per-behavior directional fidelity\n(green = higher cosine shift)",
        fontsize=9,
        pad=8,
    )
    plt.colorbar(im, ax=ax, label="Mean cosine shift", shrink=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading efficacy results...")
    df = load_all_results()

    log.info("Aggregating across behaviors...")
    agg = aggregate(df)
    agg.to_csv(OUT_DIR / "aggregate_table.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "aggregate_table.csv")

    gain = compute_gain_table(df)
    gain.to_csv(OUT_DIR / "gain_table.csv", index=False)
    log.info("Saved: %s", OUT_DIR / "gain_table.csv")

    log.info("\n=== Aggregate mean cosine shift (mean ± CI95 over 5 behaviors) ===")
    models_order = list(MODEL_DIRS.values())
    for model in models_order:
        log.info("  %s:", model)
        model_agg = agg[agg["model"] == model].set_index("method")
        for method in METHOD_ORDER:
            if method in model_agg.index:
                row = model_agg.loc[method]
                log.info(
                    "    %-25s  %.5f ± %.5f",
                    METHOD_LABELS[method],
                    row["mcs_mean"],
                    row["ci95"],
                )

    log.info("\n=== K-calibrated gain over uncalibrated ===")
    for _, row in gain.iterrows():
        sign = "+" if row["abs_gain"] >= 0 else ""
        log.info(
            "  %-15s  abs_gain=%s%.5f  rel_gain=%s%.1f%%  vs_raw=%s%.5f",
            row["model"],
            sign,
            row["abs_gain"],
            sign,
            row["rel_gain_pct"],
            "+" if row["vs_raw_abs"] >= 0 else "",
            row["vs_raw_abs"],
        )

    log.info("Building LaTeX table...")
    latex = build_latex_table(agg, gain)
    tex_path = OUT_DIR / "latex_table.tex"
    tex_path.write_text(latex, encoding="utf-8")
    log.info("Saved LaTeX table: %s", tex_path)

    log.info("Generating figures...")
    plot_directional_fidelity(df, FIG_DIR / "fig_appendix_b_directional_fidelity.pdf")
    plot_behavior_heatmap(df, FIG_DIR / "fig_appendix_b_heatmap.pdf")

    log.info("Done. Outputs in %s and %s", OUT_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
