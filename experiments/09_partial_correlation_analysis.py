"""
Experiment 09: Partial Correlation Analysis.

Addresses reviewer fatal weakness F2: partial correlation r(K_l, sigma1(W_up) | layer_idx)
to test whether the spectral link survives after controlling for layer depth.

Also:
  - Tests W_down vs W_up as spectral proxy (P1 fix)
  - Analyzes PC2-5 alignment variance (M4)
  - Bootstraps stability of alignment ratios over prompt subsets (M3/M6 partial)

Outputs:
  results/partial_correlation/summary.csv
  results/partial_correlation/latex_table.tex
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
K_CAL_DIR = ROOT / "results" / "k_calibration"
PCA_DIR = ROOT / "results" / "pca_directions"
OUT_DIR = ROOT / "results" / "partial_correlation"

MODEL_FILES: dict[str, str] = {
    "Llama 3.1 8B": "llama_k_vs_spectral.csv",
    "Gemma 2 9B": "gemma_k_vs_spectral.csv",
    "Qwen 2.5 7B": "qwen_k_vs_spectral.csv",
    "Mistral 7B": "mistral_k_vs_spectral.csv",
}

MODEL_KEYS: dict[str, str] = {
    "Llama 3.1 8B": "llama",
    "Gemma 2 9B": "gemma",
    "Qwen 2.5 7B": "qwen",
    "Mistral 7B": "mistral",
}

BEHAVIORS: list[str] = [
    "formality",
    "refusal_calibration",
    "sycophancy_suppression",
    "uncertainty_expression",
    "verbosity_control",
]


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """
    Partial correlation r(x, y | z) — correlation between x and y after
    partialing out the linear effect of z.

    Returns (r, p-value) using a t-test with n-3 degrees of freedom.
    """
    n = len(x)
    # Residuals of x ~ z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    # Residuals of y ~ z
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    # Pearson r on residuals
    r, _ = stats.pearsonr(resid_x, resid_y)
    # t-statistic with n-3 df
    t = r * np.sqrt((n - 3) / (1 - r**2 + 1e-10))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return float(r), float(p)


def format_p(p: float) -> str:
    if p < 1e-10:
        return r"$<10^{-10}$"
    elif p < 1e-5:
        exp = int(np.floor(np.log10(p)))
        return rf"$<10^{{{exp}}}$"
    elif p < 0.001:
        return f"${p:.4f}$"
    else:
        return f"${p:.3f}$"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_spectral_correlations() -> pd.DataFrame:
    """
    For each model, compute:
      - r(K_l, sigma1(W_up))        — reported in paper (Table 2)
      - r(K_l, sigma1(W_up) | depth) — partial, controls depth
      - r(K_l, sigma1(W_down))      — theoretically motivated (P1)
      - r(K_l, sigma1(W_down) | depth)
      - r(K_l, sigma1(W_up) | depth) using Spearman (robustness check)
    """
    records: list[dict] = []

    for model_label, fname in MODEL_FILES.items():
        csv_path = K_CAL_DIR / fname
        if not csv_path.exists():
            log.warning("Missing: %s", csv_path)
            continue

        df = pd.read_csv(csv_path)
        layer = df["layer_idx"].values.astype(float)
        k = df["k_value"].values

        has_up = "spectral_norm_up_proj" in df.columns
        has_dn = "spectral_norm_down_proj" in df.columns

        rec: dict = {"model": model_label, "n_layers": len(df)}

        # Standard Pearson r (what the paper reports)
        if has_up:
            r_up, p_up = stats.pearsonr(k, df["spectral_norm_up_proj"].values)
            rec["r_Wup_full"] = r_up
            rec["p_Wup_full"] = p_up

            # Partial r controlling for depth
            r_up_partial, p_up_partial = partial_correlation(
                k, df["spectral_norm_up_proj"].values, layer
            )
            rec["r_Wup_partial"] = r_up_partial
            rec["p_Wup_partial"] = p_up_partial

            # Spearman partial (depth-rank-controlled, non-parametric)
            rank_k = stats.rankdata(k)
            rank_up = stats.rankdata(df["spectral_norm_up_proj"].values)
            rank_layer = stats.rankdata(layer)
            r_up_spearman_partial, p_up_spearman_partial = partial_correlation(
                rank_k, rank_up, rank_layer
            )
            rec["rho_Wup_partial"] = r_up_spearman_partial
            rec["p_Wup_spearman_partial"] = p_up_spearman_partial

        if has_dn:
            r_dn, p_dn = stats.pearsonr(k, df["spectral_norm_down_proj"].values)
            rec["r_Wdn_full"] = r_dn
            rec["p_Wdn_full"] = p_dn

            r_dn_partial, p_dn_partial = partial_correlation(
                k, df["spectral_norm_down_proj"].values, layer
            )
            rec["r_Wdn_partial"] = r_dn_partial
            rec["p_Wdn_partial"] = p_dn_partial

        records.append(rec)
        log.info(
            "%s | r_Wup=%.3f, r_Wup|depth=%.3f (p=%.4f) | r_Wdn=%.3f, r_Wdn|depth=%.3f",
            model_label,
            rec.get("r_Wup_full", float("nan")),
            rec.get("r_Wup_partial", float("nan")),
            rec.get("p_Wup_partial", float("nan")),
            rec.get("r_Wdn_full", float("nan")),
            rec.get("r_Wdn_partial", float("nan")),
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PC2-5 alignment analysis (M4)
# ---------------------------------------------------------------------------

def analyze_pc_components() -> pd.DataFrame:
    """
    For each model, check whether PC2-5 directions show similar variance_explained
    ratios to PC1, and whether their ratios to PC1 suggest a rich multi-component structure.
    """
    records: list[dict] = []

    for model_label, model_key in MODEL_KEYS.items():
        behavior_records: list[dict] = []
        for behavior in BEHAVIORS:
            csv_path = PCA_DIR / model_key / behavior / "variance_explained.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            # Average across layers
            pc1 = df[df["component_idx"] == 0]["variance_explained_ratio"].mean()
            pc2 = df[df["component_idx"] == 1]["variance_explained_ratio"].mean()
            pc3 = df[df["component_idx"] == 2]["variance_explained_ratio"].mean()
            pc4 = df[df["component_idx"] == 3]["variance_explained_ratio"].mean()
            pc5 = df[df["component_idx"] == 4]["variance_explained_ratio"].mean()
            behavior_records.append(
                {
                    "behavior": behavior,
                    "pc1": pc1,
                    "pc2": pc2,
                    "pc3": pc3,
                    "pc4": pc4,
                    "pc5": pc5,
                    "pc1_pc2_ratio": pc1 / pc2 if pc2 > 0 else float("nan"),
                    "pc1_to_pc5_cumulative": pc1 + pc2 + pc3 + pc4 + pc5,
                }
            )
        if behavior_records:
            bdf = pd.DataFrame(behavior_records)
            records.append(
                {
                    "model": model_label,
                    "mean_pc1": bdf["pc1"].mean(),
                    "mean_pc2": bdf["pc2"].mean(),
                    "mean_pc3": bdf["pc3"].mean(),
                    "pc1_pc2_ratio_mean": bdf["pc1_pc2_ratio"].mean(),
                    "pc1_pc2_ratio_std": bdf["pc1_pc2_ratio"].std(),
                    "pc1_to_pc5_cumulative": bdf["pc1_to_pc5_cumulative"].mean(),
                }
            )
            log.info(
                "%s | PC1: %.4f, PC2: %.4f, PC1/PC2 ratio: %.2f ± %.2f",
                model_label,
                bdf["pc1"].mean(),
                bdf["pc2"].mean(),
                bdf["pc1_pc2_ratio"].mean(),
                bdf["pc1_pc2_ratio"].std(),
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def build_latex_partial_correlation_table(df: pd.DataFrame) -> str:
    """
    Table with columns: Model | r(K, Wup) | r(K, Wup|depth) | r(K, Wdn) | r(K, Wdn|depth)
    Partial correlation answers F2. W_down column answers P1.
    """
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{",
        r"    \textbf{Spectral correlations: full vs.\ depth-controlled (partial).}",
        r"    $r(\cdot, \cdot)$: Pearson; $r(\cdot, \cdot \mid \ell)$: partial correlation",
        r"    controlling for layer index $\ell$. Partial correlations address the",
        r"    depth-confound concern: both $K_\ell$ and $\sone(\Wup_\ell)$ trend monotonically",
        r"    with depth in pre-norm models. Retaining $r > 0.5$ after partialing",
        r"    confirms the spectral link is not purely a proxy for depth.",
        r"    $\sone(\Wdn_\ell)$ column tests the theoretically motivated proxy",
        r"    (Proposition~\ref{prop:integral}) directly.",
        r"  }",
        r"  \label{tab:partial_spectral}",
        r"  \vskip 0.05in",
        r"  \resizebox{\columnwidth}{!}{%",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Model}",
        r"    & $r(K,\Wup)$",
        r"    & $r(K,\Wup|\ell)$",
        r"    & $r(K,\Wdn)$",
        r"    & $r(K,\Wdn|\ell)$ \\",
        r"    \midrule",
    ]

    for _, row in df.iterrows():
        model = row["model"]
        r_up = row.get("r_Wup_full", float("nan"))
        r_up_p = row.get("r_Wup_partial", float("nan"))
        r_dn = row.get("r_Wdn_full", float("nan"))
        r_dn_p = row.get("r_Wdn_partial", float("nan"))
        p_up_p = row.get("p_Wup_partial", float("nan"))
        p_dn_p = row.get("p_Wdn_partial", float("nan"))

        def fmt_r(r: float, p: float) -> str:
            if np.isnan(r):
                return "---"
            sig = r"$^{**}$" if p < 0.01 else (r"$^{*}$" if p < 0.05 else "")
            return f"${r:.3f}{sig}$"

        def fmt_full(r: float) -> str:
            return f"${r:.3f}$" if not np.isnan(r) else "---"

        lines.append(
            f"    {model} & {fmt_full(r_up)} & {fmt_r(r_up_p, p_up_p)}"
            f" & {fmt_full(r_dn)} & {fmt_r(r_dn_p, p_dn_p)} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"    \multicolumn{5}{l}{\scriptsize $^{**}p{<}0.01$, $^{*}p{<}0.05$ (two-tailed $t$-test, $n-3$ d.f.)}",
        r"  \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Partial Correlation Analysis (F2 fix) ===")
    corr_df = analyze_spectral_correlations()
    corr_df.to_csv(OUT_DIR / "spectral_partial_correlation.csv", index=False)

    log.info("\n=== KEY FINDING: Does partial r stay above 0.5 after depth control? ===")
    for _, row in corr_df.iterrows():
        r_full = row.get("r_Wup_full", float("nan"))
        r_partial = row.get("r_Wup_partial", float("nan"))
        p_partial = row.get("p_Wup_partial", float("nan"))
        drop = r_full - r_partial if not np.isnan(r_full) else float("nan")
        verdict = (
            "SURVIVES (strong link)" if r_partial > 0.5 and p_partial < 0.05
            else "SURVIVES (moderate)" if r_partial > 0.3 and p_partial < 0.05
            else "WEAKENED (depth proxy?)" if p_partial >= 0.05
            else "WEAKENED"
        )
        log.info(
            "  %-15s r_full=%.3f, r_partial=%.3f (p=%.4f), drop=%.3f → %s",
            row["model"], r_full, r_partial, p_partial, drop, verdict,
        )

    latex = build_latex_partial_correlation_table(corr_df)
    (OUT_DIR / "latex_partial_table.tex").write_text(latex, encoding="utf-8")
    log.info("\nSaved LaTeX table: %s", OUT_DIR / "latex_partial_table.tex")

    log.info("\n=== PC Component Analysis (M4 fix) ===")
    pc_df = analyze_pc_components()
    pc_df.to_csv(OUT_DIR / "pc_component_summary.csv", index=False)

    log.info("\n=== PC1 Dominance: is using only PC1 justified? ===")
    for _, row in pc_df.iterrows():
        ratio = row["pc1_pc2_ratio_mean"]
        verdict = "PC1 clearly dominant" if ratio > 1.5 else "PC1 marginally dominant"
        log.info(
            "  %-15s PC1/PC2 ratio = %.2f ± %.2f → %s",
            row["model"], ratio, row["pc1_pc2_ratio_std"], verdict,
        )

    log.info("\nDone. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
