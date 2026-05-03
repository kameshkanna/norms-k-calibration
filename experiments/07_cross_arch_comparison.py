"""
experiments/07_cross_arch_comparison.py

Cross-Architecture Comparison — evidence for universal behavioral geometry.

For each behavior and each pair of models the script:
  1. Loads saved PCA directions (or contrastive diffs) for every model.
  2. Computes Centered Kernel Alignment (CKA) between the contrastive diff
     matrices at matching relative layer depths (0%, 25%, 50%, 75%, 100%).
  3. Computes principal-angle cosine similarity between the PCA subspaces at
     the same depth fractions.

Outputs (per behavior):
  results/cross_arch/{behavior}/cka_matrix.csv
    4×4 model similarity matrix (averaged over layer fractions)
  results/cross_arch/{behavior}/layer_depth_similarity.csv
    Columns: layer_fraction, model_pair, cka_score, subspace_cosine_sim
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("07_cross_arch_comparison")

# ---------------------------------------------------------------------------
# Default model list (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_MODELS: List[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Relative depth fractions at which we compare representations.
DEPTH_FRACTIONS: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


# ---------------------------------------------------------------------------
# Centered Kernel Alignment
# ---------------------------------------------------------------------------

def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute the unbiased HSIC estimator (Kornblith et al., 2019).

    The biased double-centering estimator approaches 1.0 when the feature
    dimension greatly exceeds the number of samples (d >> n), which is
    the typical regime for PCA direction matrices (n ≤ 60, d ≥ 3584).
    This unbiased estimator corrects for that artifact by zeroing out the
    kernel diagonal before computing the statistic.

    Definition (Eq. 3 in Kornblith et al. 2019)::

        HSIC_0(K, L) = 1 / (n(n-3)) * [
            sum_{i≠j} K̃_ij L̃_ij
            + (sum_{i≠j} K̃_ij)(sum_{i≠j} L̃_ij) / ((n-1)(n-2))
            − (2/(n-2)) * tr(K̃ L̃)
        ]

    where K̃ is K with diagonal set to zero.

    Args:
        K: Symmetric kernel matrix of shape ``[n, n]``, ``n ≥ 4``.
        L: Symmetric kernel matrix of shape ``[n, n]``.

    Returns:
        Scalar tensor.  May be negative for small n; clamp before sqrt.

    Raises:
        ValueError: If ``n < 4``.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(
            f"Unbiased HSIC requires n ≥ 4 samples; got {n}. "
            "Increase the number of contrastive pairs or fall back to biased CKA."
        )

    # Zero out diagonals in-place copies (do not mutate the caller's tensors).
    K_tilde = K.clone()
    K_tilde.fill_diagonal_(0.0)
    L_tilde = L.clone()
    L_tilde.fill_diagonal_(0.0)

    # Term 1: Frobenius inner product of zeroed-diagonal kernels.
    # Equivalent to sum_{i≠j} K̃_ij L̃_ij because diagonals are zero.
    term1 = (K_tilde * L_tilde).sum()

    # Term 2: product of marginal sums, normalised.
    sum_K = K_tilde.sum()
    sum_L = L_tilde.sum()
    term2 = sum_K * sum_L / ((n - 1) * (n - 2))

    # Term 3: 2/(n-2) * tr(K̃ L̃).  tr(AB) = Σ_i (AB)_ii.
    term3 = (2.0 / (n - 2)) * torch.trace(K_tilde @ L_tilde)

    return (term1 + term2 - term3) / (n * (n - 3))


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA with the unbiased HSIC estimator.

    Uses the Kornblith et al. (2019) unbiased estimator, which is robust to
    the high-dimension / low-sample regime that arises when comparing PCA
    direction matrices (n ≈ 5 components, d ≥ 3584 hidden dimensions).

    Linear CKA is defined as:
        CKA(X, Y) = HSIC_0(XX^T, YY^T) / sqrt(HSIC_0(XX^T, XX^T) * HSIC_0(YY^T, YY^T))

    Args:
        X: Representation matrix of shape ``[n_samples, d1]``.
        Y: Representation matrix of shape ``[n_samples, d2]``.

    Returns:
        CKA scalar.  Returns ``0.0`` when the denominator is negligible or when
        ``n < 4`` (falls back gracefully).

    Raises:
        ValueError: If ``X`` and ``Y`` have different numbers of rows.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples; "
            f"got {X.shape[0]} vs {Y.shape[0]}."
        )

    n = X.shape[0]
    if n < 4:
        logger.warning(
            "cka: n=%d < 4; unbiased HSIC requires n ≥ 4.  Returning 0.0.", n
        )
        return 0.0

    # Centre the representations (mean subtraction per feature).
    X_c = X.float() - X.float().mean(dim=0, keepdim=True)
    Y_c = Y.float() - Y.float().mean(dim=0, keepdim=True)

    K = X_c @ X_c.T   # [n, n]
    L = Y_c @ Y_c.T   # [n, n]

    hsic_kl = _hsic_unbiased(K, L)
    hsic_kk = _hsic_unbiased(K, K)
    hsic_ll = _hsic_unbiased(L, L)

    # Guard against negative denominator terms (can occur for very small n).
    kk_val = hsic_kk.item()
    ll_val = hsic_ll.item()
    if kk_val <= 0.0 or ll_val <= 0.0:
        logger.warning(
            "cka: non-positive HSIC diagonal (HSIC_kk=%.4e, HSIC_ll=%.4e); "
            "insufficient samples for reliable estimate. Returning 0.0.",
            kk_val,
            ll_val,
        )
        return 0.0

    denominator = torch.sqrt(hsic_kk * hsic_ll)
    if denominator.abs().item() < 1e-12:
        return 0.0

    return (hsic_kl / denominator).item()


# ---------------------------------------------------------------------------
# Subspace similarity via principal angles
# ---------------------------------------------------------------------------

@torch.no_grad()
def principal_angle_cosine(
    A: torch.Tensor,
    B: torch.Tensor,
) -> float:
    """Compute the mean cosine of principal angles between two subspaces.

    Args:
        A: Matrix whose columns (or rows) span subspace A, shape ``[n, k1]``.
        B: Matrix whose columns (or rows) span subspace B, shape ``[n, k2]``.
            Assumes k1 == k2 == k (same number of components).

    Returns:
        Mean cosine of the k principal angles, in ``[0, 1]``.
    """
    # Orthonormalise via QR decomposition.
    Q_A, _ = torch.linalg.qr(A.float().T)  # [n, k1] after transpose → Q_A: [k1, n] → [n, k1]
    Q_B, _ = torch.linalg.qr(B.float().T)

    # Q_A is [n, k1], Q_B is [n, k2].
    # Cosines of principal angles = singular values of Q_A^T @ Q_B.
    M = Q_A.T @ Q_B   # [k1, k2]
    sigmas = torch.linalg.svdvals(M)  # singular values in decreasing order
    # Clamp to [-1, 1] for numerical safety before taking arccos.
    sigmas = sigmas.clamp(-1.0, 1.0)
    return sigmas.mean().item()


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _model_slug(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe slug.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Slug string safe for use as a directory name.
    """
    return model_id.replace("/", "__")


# Map from HuggingFace model ID to the short key used by experiment 02
# as the directory name under results/pca_directions/.
_HF_ID_TO_SHORT_KEY: Dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "google/gemma-2-9b-it": "gemma",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
}


def _resolve_pca_dir(results_dir: Path, model_id: str, behavior: str) -> Path:
    """Return the directory containing saved PCA directions.

    Tries the short model key (e.g. ``"llama"``) used by experiment 02 first,
    then falls back to the slugified HF ID.  Returns the first directory that
    contains a ``directions.pt`` file, or the primary candidate if none exists.

    Args:
        results_dir: Root results directory.
        model_id: HuggingFace model identifier.
        behavior: Behavior name.

    Returns:
        Path to the best candidate directory.
    """
    candidates: List[Path] = []
    short_key = _HF_ID_TO_SHORT_KEY.get(model_id)
    if short_key:
        candidates.append(results_dir / "pca_directions" / short_key / behavior)
    candidates.append(results_dir / "pca_directions" / _model_slug(model_id) / behavior)
    for candidate in candidates:
        if (candidate / "directions.pt").exists():
            return candidate
    return candidates[0]


def _extract_direction_tensor(v: object) -> Optional[torch.Tensor]:
    """Extract a float32 component tensor from a direction value.

    Handles both raw ``torch.Tensor`` and ``BehavioralDirections`` dataclass
    objects (produced by experiment 02), transparently returning the underlying
    ``[n_components, hidden]`` tensor in both cases.

    Args:
        v: A direction value from a loaded ``directions.pt`` dict.

    Returns:
        Float32 tensor of shape ``[n_components, hidden]``, or ``None`` if the
        type is unrecognised.
    """
    if isinstance(v, torch.Tensor):
        return v.float()
    if hasattr(v, "components") and isinstance(v.components, torch.Tensor):
        return v.components.float()
    return None


def _key_to_layer_int(k: object) -> Optional[int]:
    """Convert a directions.pt key to an integer layer index.

    Handles both plain integer keys and string module-path keys such as
    ``"model.layers.8"`` (produced by experiment 02).

    Args:
        k: Dict key from a loaded ``directions.pt`` file.

    Returns:
        Integer layer index, or ``None`` if conversion fails.
    """
    if isinstance(k, int):
        return k
    if isinstance(k, str):
        try:
            return int(k.split(".")[-1])
        except ValueError:
            return None
    return None


def load_pca_directions(
    results_dir: Path,
    model_id: str,
    behavior: str,
) -> Optional[Dict[int, torch.Tensor]]:
    """Load saved PCA direction tensors for a model/behavior pair.

    Normalises the loaded file to ``Dict[int, Tensor[n_components, hidden]]``
    regardless of the on-disk format (experiment 02 saves
    ``Dict[str, BehavioralDirections]``; other formats are also supported).

    Args:
        results_dir: Root results directory.
        model_id: HuggingFace model identifier.
        behavior: Behavior name.

    Returns:
        Dict mapping integer layer index to PCA direction tensor
        ``[n_components, hidden]``, or ``None`` if the file is absent or
        cannot be parsed.
    """
    pca_dir = _resolve_pca_dir(results_dir, model_id, behavior)
    directions_path = pca_dir / "directions.pt"

    if not directions_path.exists():
        logger.warning("Directions not found (tried short key and slug): %s", directions_path)
        return None

    payload = torch.load(directions_path, map_location="cpu")

    if isinstance(payload, dict):
        result: Dict[int, torch.Tensor] = {}
        for raw_key, raw_val in payload.items():
            idx = _key_to_layer_int(raw_key)
            if idx is None:
                logger.debug("Skipping unrecognised key '%s' in directions.pt.", raw_key)
                continue
            tensor = _extract_direction_tensor(raw_val)
            if tensor is None:
                logger.debug("Skipping key '%s': value type %s not recognised.", raw_key, type(raw_val))
                continue
            result[idx] = tensor
        return result if result else None

    if isinstance(payload, torch.Tensor) and payload.dim() == 3:
        return {i: payload[i].float() for i in range(payload.shape[0])}

    logger.error(
        "Unrecognised directions.pt format for %s / %s: %s",
        model_id,
        behavior,
        type(payload),
    )
    return None


def load_raw_diffs(
    results_dir: Path,
    model_id: str,
    behavior: str,
) -> Optional[Dict[int, torch.Tensor]]:
    """Load saved raw contrastive diff matrices for a model/behavior pair.

    Raw diffs are expected as a .pt file with the same structure as directions.pt
    but containing the per-sample diffs before PCA, i.e. shape ``[n_samples, hidden]``
    per layer.

    Args:
        results_dir: Root results directory.
        model_id: HuggingFace model identifier.
        behavior: Behavior name.

    Returns:
        Dict mapping layer index to diff matrix ``[n_samples, hidden]``,
        or ``None`` if absent.
    """
    pca_dir = _resolve_pca_dir(results_dir, model_id, behavior)
    diffs_path = pca_dir / "raw_diffs.pt"

    if not diffs_path.exists():
        return None

    payload = torch.load(diffs_path, map_location="cpu")
    if isinstance(payload, dict):
        result: Dict[int, torch.Tensor] = {}
        for raw_key, raw_val in payload.items():
            idx = _key_to_layer_int(raw_key)
            if idx is None:
                logger.debug("Skipping unrecognised key '%s' in raw_diffs.pt.", raw_key)
                continue
            if isinstance(raw_val, torch.Tensor):
                result[idx] = raw_val.float()
        return result if result else None
    logger.warning("Unexpected format for raw_diffs.pt: %s", type(payload))
    return None


def get_direction_at_fraction(
    directions_by_layer: Dict[int, torch.Tensor],
    fraction: float,
) -> Tuple[Optional[torch.Tensor], int]:
    """Return the direction tensor closest to a given relative depth fraction.

    Args:
        directions_by_layer: Dict mapping layer index to direction tensor.
        fraction: Relative depth in ``[0, 1]``.

    Returns:
        Tuple of (tensor or None, resolved_layer_index).
    """
    if not directions_by_layer:
        return None, -1
    n_layers = max(directions_by_layer.keys()) + 1
    target_idx = int(round(fraction * (n_layers - 1)))
    # Find the nearest available layer index.
    available = sorted(directions_by_layer.keys())
    nearest = min(available, key=lambda x: abs(x - target_idx))
    return directions_by_layer[nearest], nearest


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_cross_arch_experiment(
    behaviors: List[str],
    model_ids: List[str],
    device: torch.device,
    output_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    """Run cross-architecture CKA comparison for all specified behaviors and models.

    Args:
        behaviors: List of behavior identifiers.
        model_ids: List of HuggingFace model identifiers.
        device: Torch device (used for tensor ops).
        output_dir: Root directory for output CSVs.
        results_dir: Root results directory where directions are stored.
        seed: Random seed (unused currently; reserved for future bootstrapping).

    Raises:
        ValueError: If fewer than two models have directions for a given behavior.
    """
    torch.manual_seed(seed)
    n_models = len(model_ids)
    model_pairs = list(itertools.combinations(range(n_models), 2))

    for behavior in tqdm(behaviors, desc="Behaviors", dynamic_ncols=True):
        logger.info("Processing behavior: %s", behavior)
        save_dir = output_dir / behavior
        save_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Load directions for all models
        # ------------------------------------------------------------------
        all_directions: Dict[str, Optional[Dict[int, torch.Tensor]]] = {}
        all_diffs: Dict[str, Optional[Dict[int, torch.Tensor]]] = {}

        for model_id in model_ids:
            dirs = load_pca_directions(results_dir, model_id, behavior)
            raw_diffs = load_raw_diffs(results_dir, model_id, behavior)
            all_directions[model_id] = dirs
            all_diffs[model_id] = raw_diffs

        available_models = [m for m in model_ids if all_directions[m] is not None]
        if len(available_models) < 2:
            logger.warning(
                "Fewer than 2 models have directions for behavior '%s'; skipping.",
                behavior,
            )
            continue

        # ------------------------------------------------------------------
        # Build short model labels for CSV columns / rows
        # ------------------------------------------------------------------
        def _label(model_id: str) -> str:
            return model_id.split("/")[-1]

        labels = [_label(m) for m in available_models]
        n_avail = len(available_models)

        # ------------------------------------------------------------------
        # Per-layer-fraction CKA and subspace cosine
        # ------------------------------------------------------------------
        layer_depth_rows: List[Dict] = []

        for fraction in DEPTH_FRACTIONS:
            for i, j in itertools.combinations(range(n_avail), 2):
                model_a = available_models[i]
                model_b = available_models[j]
                pair_label = f"{labels[i]}_vs_{labels[j]}"

                dirs_a = all_directions[model_a]
                dirs_b = all_directions[model_b]
                diffs_a = all_diffs[model_a]
                diffs_b = all_diffs[model_b]

                # Prefer raw diffs for CKA (they are sample matrices);
                # fall back to PCA direction matrices.
                repr_a_src = diffs_a if diffs_a else dirs_a
                repr_b_src = diffs_b if diffs_b else dirs_b

                repr_a, layer_a = get_direction_at_fraction(repr_a_src, fraction)
                repr_b, layer_b = get_direction_at_fraction(repr_b_src, fraction)

                if repr_a is None or repr_b is None:
                    logger.debug(
                        "Missing representation for %s at fraction=%.2f; skipping.",
                        pair_label,
                        fraction,
                    )
                    continue

                # Ensure both matrices have the same sample count for CKA.
                # They may differ in n_samples; align by taking the minimum.
                n_a, n_b = repr_a.shape[0], repr_b.shape[0]
                n_common = min(n_a, n_b)
                repr_a_aligned = repr_a[:n_common].float().to(device)
                repr_b_aligned = repr_b[:n_common].float().to(device)

                # Reduce to same hidden dim via PCA when architectures differ.
                if repr_a_aligned.shape[1] != repr_b_aligned.shape[1]:
                    min_dim = min(repr_a_aligned.shape[1], repr_b_aligned.shape[1])
                    repr_a_aligned = repr_a_aligned[:, :min_dim]
                    repr_b_aligned = repr_b_aligned[:, :min_dim]

                cka_score = cka(repr_a_aligned, repr_b_aligned)

                # Subspace cosine: use PCA direction matrices (components × hidden).
                dirs_a_frac, _ = get_direction_at_fraction(dirs_a, fraction)
                dirs_b_frac, _ = get_direction_at_fraction(dirs_b, fraction)
                subspace_cos = float("nan")
                if dirs_a_frac is not None and dirs_b_frac is not None:
                    # Transpose to [hidden, n_components] for QR-based principal angles.
                    A = dirs_a_frac.float().T  # [hidden, k]
                    B = dirs_b_frac.float().T  # [hidden, k]
                    if A.shape[0] != B.shape[0]:
                        min_h = min(A.shape[0], B.shape[0])
                        A = A[:min_h]
                        B = B[:min_h]
                    try:
                        subspace_cos = principal_angle_cosine(A, B)
                    except (torch.linalg.LinAlgError, ValueError) as exc:
                        logger.warning(
                            "Principal angle computation failed for %s at %.2f: %s",
                            pair_label,
                            fraction,
                            exc,
                        )

                layer_depth_rows.append(
                    {
                        "layer_fraction": fraction,
                        "model_pair": pair_label,
                        "model_a": labels[i],
                        "model_b": labels[j],
                        "layer_a_idx": layer_a,
                        "layer_b_idx": layer_b,
                        "cka_score": cka_score,
                        "subspace_cosine_sim": subspace_cos,
                    }
                )

                del repr_a_aligned, repr_b_aligned
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Build 4×4 CKA matrix (averaged over all depth fractions)
        # ------------------------------------------------------------------
        layer_depth_df = pd.DataFrame(layer_depth_rows)

        cka_matrix = np.full((n_avail, n_avail), fill_value=np.nan)
        np.fill_diagonal(cka_matrix, 1.0)  # self-similarity = 1

        if not layer_depth_df.empty:
            avg_cka = (
                layer_depth_df.groupby(["model_a", "model_b"])["cka_score"]
                .mean()
                .reset_index()
            )
            for _, row in avg_cka.iterrows():
                if row["model_a"] in labels and row["model_b"] in labels:
                    i = labels.index(row["model_a"])
                    j = labels.index(row["model_b"])
                    cka_matrix[i, j] = row["cka_score"]
                    cka_matrix[j, i] = row["cka_score"]  # symmetric

        cka_df = pd.DataFrame(cka_matrix, index=labels, columns=labels)

        # ------------------------------------------------------------------
        # Persist results
        # ------------------------------------------------------------------
        cka_csv_path = save_dir / "cka_matrix.csv"
        depth_csv_path = save_dir / "layer_depth_similarity.csv"

        cka_df.to_csv(cka_csv_path)
        layer_depth_df.to_csv(depth_csv_path, index=False)

        logger.info("Saved CKA matrix            → %s", cka_csv_path)
        logger.info("Saved layer-depth similarity → %s", depth_csv_path)
        if not layer_depth_df.empty:
            mean_cka = layer_depth_df["cka_score"].mean()
            mean_subcos = layer_depth_df["subspace_cosine_sim"].dropna().mean()
            logger.info(
                "Behavior '%s' summary — mean CKA: %.4f | mean subspace cosine: %.4f",
                behavior,
                mean_cka,
                mean_subcos,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 07: Cross-Architecture Comparison — CKA-based evidence "
            "for universal behavioral geometry across LLM architectures."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=["sycophancy_suppression", "uncertainty_expression"],
        help="One or more behavior identifiers.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        dest="models",
        help="HuggingFace model identifiers to compare.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cross_arch"),
        help="Root directory for output CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (contains pca_directions/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 07."""
    args = _parse_args()
    device = torch.device(args.device)

    run_cross_arch_experiment(
        behaviors=args.behaviors,
        model_ids=args.models,
        device=device,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
