"""
experiments/04_permutation_invariance.py

THE KEY EXPERIMENT: Demonstrate that PCA behavioral directions are invariant
under neuron permutations, providing empirical evidence that they occupy
weight-space symmetry-invariant subspaces.

For each (model, behavior) pair the script:
  1. Loads pre-fitted PCA directions from script 02 (or regenerates them).
  2. For each of ``n_permutations`` random seeds:
     a. Selects a random subset of layer indices (``permute_fraction`` of total).
     b. Creates a functionally equivalent permuted model via ``apply_neuron_permutation``.
     c. Re-runs contrastive diff extraction on the SAME training prompts.
     d. Re-fits PCADirector on permuted activations.
     e. Computes per-layer subspace similarity via the principal-angle method:
            M = dirs_a @ dirs_b.T
            S = svdvals(M)         # principal cosines
            score = mean(S)
  3. Aggregates all scores and saves per-permutation CSVs and summary JSONs.

**Expected result (paper claim):** Mean subspace cosine similarity > 0.85
across all layers and permutations.

Outputs (per model × behavior)
-------------------------------
{output_dir}/{model}/{behavior}/invariance_scores.csv
    Columns: permutation_seed, layer_idx, layer_name, subspace_cosine_sim,
    n_layers_permuted, model_key, behavior.
{output_dir}/{model}/{behavior}/summary.json
    Keys: mean_cosine_sim, std_cosine_sim, min_cosine_sim, max_cosine_sim,
    n_permutations, n_layers, model_key, behavior, claim_supported (>0.85).

Usage
-----
python experiments/04_permutation_invariance.py \\
    --model llama --behavior sycophancy_suppression \\
    --n-permutations 5 --permute-fraction 0.5 --device cuda
python experiments/04_permutation_invariance.py --model all --behavior all
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed

from activation_baking.model_utils import (
    ModelInfo,
    detect_model_info,
    apply_neuron_permutation,
)
from activation_baking.extractor import ActivationExtractor
from activation_baking.pca_director import PCADirector, BehavioralDirections

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_BEHAVIORS: List[str] = [
    "sycophancy_suppression",
    "refusal_calibration",
    "verbosity_control",
    "formality",
    "uncertainty_expression",
]

# Minimum mean cosine similarity to declare permutation invariance (paper claim).
INVARIANCE_THRESHOLD: float = 0.85

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(tag: str, output_dir: Path) -> None:
    """Configure root logger to write to console and a timestamped file.

    Args:
        tag: Short tag used in the log filename (e.g. ``"llama_sycophancy"``).
        output_dir: Base output directory; logs go under ``results/logs/``.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"04_permutation_invariance_{tag}_{timestamp}.log"

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
    """Set all random seeds for full reproducibility.

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
    """Resolve a device string, falling back to CPU if CUDA is unavailable.

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
    """Return current GPU allocation in GiB, or 0.0 for CPU.

    Args:
        device: The torch device to query.

    Returns:
        Allocated memory in GiB.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Subspace similarity
# ---------------------------------------------------------------------------


def compute_subspace_similarity(
    dirs_a: torch.Tensor,
    dirs_b: torch.Tensor,
) -> float:
    """Compute mean cosine of principal angles between two linear subspaces.

    Given two sets of unit-norm basis vectors the function computes the
    cross-Gram matrix M = dirs_a @ dirs_b.T and returns the mean of the
    singular values, which are the cosines of the principal angles between
    the two subspaces.

    A score of 1.0 indicates identical subspaces; 0.0 indicates completely
    orthogonal subspaces.

    Algorithm::

        M = dirs_a @ dirs_b.T        # [n_a, n_b]
        S = svdvals(M)                # principal angle cosines ∈ [0, 1]
        score = mean(S)

    Args:
        dirs_a: Tensor of shape ``[n_components_a, hidden_size]`` containing
            unit-norm direction vectors for subspace A.
        dirs_b: Tensor of shape ``[n_components_b, hidden_size]`` containing
            unit-norm direction vectors for subspace B.

    Returns:
        Mean of the singular values of ``dirs_a @ dirs_b.T``, clamped to [0, 1].

    Raises:
        ValueError: If either tensor is not 2-D or hidden_size dimensions mismatch.
    """
    if dirs_a.ndim != 2 or dirs_b.ndim != 2:
        raise ValueError(
            f"Both direction tensors must be 2-D. "
            f"Got dirs_a.shape={tuple(dirs_a.shape)}, dirs_b.shape={tuple(dirs_b.shape)}."
        )
    if dirs_a.shape[1] != dirs_b.shape[1]:
        raise ValueError(
            f"Hidden size mismatch: dirs_a has {dirs_a.shape[1]} but "
            f"dirs_b has {dirs_b.shape[1]}."
        )

    # Compute in float32 on CPU for numerical stability
    a = dirs_a.detach().float().cpu()
    b = dirs_b.detach().float().cpu()

    cross_gram: torch.Tensor = a @ b.T  # [n_a, n_b]
    singular_values = torch.linalg.svdvals(cross_gram)  # [min(n_a, n_b)]
    singular_values = singular_values.clamp(0.0, 1.0)
    return singular_values.mean().item()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_contrastive_pairs(behavior: str, data_root: Path) -> Tuple[List[str], List[str]]:
    """Load positive and negative prompts from a JSONL behavior file.

    Args:
        behavior: Behavior name matching a file under ``data/behaviors/``.
        data_root: Root data directory.

    Returns:
        Tuple of (positives, negatives) with equal-length lists.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If required keys are missing or lists have unequal length.
    """
    jsonl_path = data_root / "behaviors" / f"{behavior}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Behavior data not found: {jsonl_path.resolve()}"
        )

    positives: List[str] = []
    negatives: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: malformed JSON — {exc}"
                ) from exc
            if "positive" not in record or "negative" not in record:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: missing 'positive' or 'negative' key."
                )
            positives.append(str(record["positive"]))
            negatives.append(str(record["negative"]))

    if len(positives) == 0:
        raise ValueError(f"No pairs found in {jsonl_path}")
    return positives, negatives


def _load_train_pairs(
    behavior: str,
    data_root: Path,
    pca_dir: Path,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Return the training split of contrastive pairs used in script 02.

    Loads the saved ``split_indices.json`` to guarantee the same pairs are
    used here as during original direction fitting — critical for a valid
    invariance test.

    Args:
        behavior: Behavior name.
        data_root: Root data directory.
        pca_dir: Directory containing ``split_indices.json``.
        seed: Fallback seed if split_indices.json is absent.

    Returns:
        Tuple of (train_positives, train_negatives).

    Raises:
        FileNotFoundError: If behavior JSONL does not exist.
    """
    logger = logging.getLogger(__name__)
    positives, negatives = _load_contrastive_pairs(behavior, data_root)

    split_path = pca_dir / "split_indices.json"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as fh:
            split_info = json.load(fh)
        train_idx: List[int] = split_info["train_indices"]
        logger.info(
            "Using saved train split (%d pairs) from %s.", len(train_idx), split_path
        )
    else:
        logger.warning(
            "split_indices.json not found at %s; recomputing 80/20 split with seed=%d.",
            split_path,
            seed,
        )
        rng = np.random.default_rng(seed)
        indices = np.arange(len(positives))
        rng.shuffle(indices)
        n_train = int(math.floor(len(positives) * 0.8))
        train_idx = indices[:n_train].tolist()

    train_pos = [positives[i] for i in train_idx]
    train_neg = [negatives[i] for i in train_idx]
    return train_pos, train_neg


# ---------------------------------------------------------------------------
# Directions loading / fallback
# ---------------------------------------------------------------------------


def _load_original_directions(
    model_key: str,
    behavior: str,
    pca_root: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_info: ModelInfo,
    device: torch.device,
    n_components: int,
    seed: int,
    data_root: Path,
) -> Dict[str, BehavioralDirections]:
    """Load pre-fitted PCA directions or regenerate them via script 02.

    Args:
        model_key: Short model key.
        behavior: Behavior name.
        pca_root: Root of the ``results/pca_directions/`` tree.
        model: Loaded eval-mode causal LM.
        tokenizer: Corresponding tokenizer.
        model_info: Structural metadata.
        device: Target device.
        n_components: Number of PCA components if regenerating.
        seed: Seed forwarded when regenerating.
        data_root: Root of ``data/`` tree.

    Returns:
        Dict[layer_name, BehavioralDirections] for the original model.
    """
    logger = logging.getLogger(__name__)
    pt_path = pca_root / model_key / behavior / "directions.pt"

    if pt_path.exists():
        logger.info("Loading original directions from %s", pt_path)
        directions: Dict[str, BehavioralDirections] = torch.load(
            str(pt_path), map_location="cpu"
        )
        return directions

    # Fallback: recompute inline using script 02
    logger.warning(
        "directions.pt not found at %s.  Running 02_contrastive_extraction.py…",
        pt_path,
    )
    script_path = Path("experiments/02_contrastive_extraction.py")
    if not script_path.exists():
        logger.warning(
            "Script 02 not found; computing directions inline without subprocess."
        )
        return _compute_directions_inline(
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            behavior=behavior,
            device=device,
            n_components=n_components,
            seed=seed,
            data_root=data_root,
            output_path=pt_path,
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--model", model_key,
        "--behavior", behavior,
        "--n-components", str(n_components),
        "--device", str(device),
        "--output-dir", str(pca_root),
        "--seed", str(seed),
    ]
    logger.info("Subprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True, check=False)
    if result.returncode != 0:
        logger.warning(
            "Subprocess exited %d; falling back to inline computation.",
            result.returncode,
        )
        return _compute_directions_inline(
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            behavior=behavior,
            device=device,
            n_components=n_components,
            seed=seed,
            data_root=data_root,
            output_path=pt_path,
        )

    if not pt_path.exists():
        raise RuntimeError(
            f"02_contrastive_extraction.py ran but {pt_path} was not produced."
        )
    directions = torch.load(str(pt_path), map_location="cpu")
    return directions


def _compute_directions_inline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_info: ModelInfo,
    behavior: str,
    device: torch.device,
    n_components: int,
    seed: int,
    data_root: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, BehavioralDirections]:
    """Compute PCA directions inline without spawning a subprocess.

    Uses the same 80/20 split seed as the original extraction to ensure
    comparability.

    Args:
        model: Loaded eval-mode causal LM.
        tokenizer: Corresponding tokenizer.
        model_info: Structural metadata.
        behavior: Behavior name.
        device: Target device.
        n_components: Number of PCA components.
        seed: Random seed for the data split.
        data_root: Root of ``data/`` tree.
        output_path: Optional path to save the resulting directions.pt.

    Returns:
        Dict[layer_name, BehavioralDirections].
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing PCA directions inline for behavior '%s'.", behavior)

    positives, negatives = _load_contrastive_pairs(behavior, data_root)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(positives))
    rng.shuffle(indices)
    n_train = int(math.floor(len(positives) * 0.8))
    train_idx = indices[:n_train].tolist()
    train_pos = [positives[i] for i in train_idx]
    train_neg = [negatives[i] for i in train_idx]

    extractor = ActivationExtractor(
        model=model, tokenizer=tokenizer, model_info=model_info, device=device
    )
    activation_diffs: Dict[str, torch.Tensor] = extractor.extract_contrastive_diffs(
        positive_prompts=train_pos,
        negative_prompts=train_neg,
        layer_names=model_info.layer_module_names,
    )

    director = PCADirector()
    directions = director.fit(activation_diffs=activation_diffs, n_components=n_components)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(directions, str(output_path))
        logger.info("Inline directions saved → %s", output_path)

    del extractor, activation_diffs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return directions


# ---------------------------------------------------------------------------
# Permutation experiment core
# ---------------------------------------------------------------------------


def _select_layer_indices_to_permute(
    num_layers: int,
    permute_fraction: float,
    rng: np.random.Generator,
) -> List[int]:
    """Randomly select a fraction of layer indices to permute.

    Args:
        num_layers: Total number of transformer layers.
        permute_fraction: Fraction of layers to select (e.g. 0.5 for half).
        rng: NumPy random generator (already seeded).

    Returns:
        Sorted list of layer indices.
    """
    n_permute = max(1, int(math.floor(num_layers * permute_fraction)))
    indices = rng.choice(num_layers, size=n_permute, replace=False)
    return sorted(indices.tolist())


def _run_single_permutation(
    permutation_seed: int,
    original_model: AutoModelForCausalLM,
    model_info: ModelInfo,
    train_pos: List[str],
    train_neg: List[str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    n_components: int,
    permute_fraction: float,
    original_directions: Dict[str, BehavioralDirections],
) -> Tuple[Dict[str, float], int]:
    """Execute one permutation trial and compute per-layer subspace similarities.

    Args:
        permutation_seed: Seed used both to select layers and to generate the
            permutation matrices in ``apply_neuron_permutation``.
        original_model: The original (unmodified) causal LM.
        model_info: Structural metadata for the model.
        train_pos: Training positive prompts (same as used for original directions).
        train_neg: Training negative prompts (same as used for original directions).
        tokenizer: Shared tokenizer.
        device: Target torch.device.
        n_components: Number of PCA components.
        permute_fraction: Fraction of layers to permute in this trial.
        original_directions: Dict of fitted directions on the original model.

    Returns:
        Tuple of:
            - Dict mapping layer_name → subspace_cosine_sim
            - int: number of layers permuted in this trial
    """
    logger = logging.getLogger(__name__)

    # ---- Select layers to permute ----
    rng = np.random.default_rng(permutation_seed)
    layer_indices = _select_layer_indices_to_permute(
        num_layers=model_info.num_layers,
        permute_fraction=permute_fraction,
        rng=rng,
    )
    logger.info(
        "Permutation seed=%d: permuting %d/%d layers: %s",
        permutation_seed,
        len(layer_indices),
        model_info.num_layers,
        layer_indices,
    )

    # ---- Apply neuron permutation (returns deep copy, does NOT modify original) ----
    permuted_model = apply_neuron_permutation(
        model=original_model,
        model_info=model_info,
        layer_indices=layer_indices,
        seed=permutation_seed,
    )
    permuted_model.to(device)
    permuted_model.eval()

    # ---- Re-extract contrastive diffs on permuted model ----
    extractor = ActivationExtractor(
        model=permuted_model, tokenizer=tokenizer, model_info=model_info, device=device
    )
    activation_diffs: Dict[str, torch.Tensor] = extractor.extract_contrastive_diffs(
        positive_prompts=train_pos,
        negative_prompts=train_neg,
        layer_names=model_info.layer_module_names,
    )

    # ---- Re-fit PCA on permuted activations ----
    director = PCADirector()
    permuted_directions: Dict[str, BehavioralDirections] = director.fit(
        activation_diffs=activation_diffs,
        n_components=n_components,
    )

    # ---- Compute per-layer subspace similarity ----
    layer_similarities: Dict[str, float] = {}
    for layer_name in model_info.layer_module_names:
        if layer_name not in original_directions or layer_name not in permuted_directions:
            logger.warning(
                "Layer '%s' missing from one of the direction sets; skipping.",
                layer_name,
            )
            continue

        orig_comps: torch.Tensor = original_directions[layer_name].components  # [k, h]
        perm_comps: torch.Tensor = permuted_directions[layer_name].components  # [k, h]

        sim_score = compute_subspace_similarity(orig_comps, perm_comps)
        layer_similarities[layer_name] = sim_score

    mean_sim = np.mean(list(layer_similarities.values())) if layer_similarities else 0.0
    logger.info(
        "  Permutation seed=%d: mean_cosine_sim=%.4f  n_layers_permuted=%d",
        permutation_seed,
        mean_sim,
        len(layer_indices),
    )

    # ---- Cleanup ----
    del permuted_model, extractor, activation_diffs, permuted_directions
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return layer_similarities, len(layer_indices)


# ---------------------------------------------------------------------------
# Per-(model, behavior) orchestration
# ---------------------------------------------------------------------------


def _run_invariance_for_behavior(
    model_key: str,
    behavior: str,
    original_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_info: ModelInfo,
    device: torch.device,
    n_permutations: int,
    permute_fraction: float,
    n_components: int,
    output_dir: Path,
    pca_root: Path,
    seed: int,
    data_root: Path,
) -> None:
    """Run all permutation trials for one (model, behavior) pair and save results.

    Args:
        model_key: Short model key (e.g. ``"llama"``).
        behavior: Behavior name.
        original_model: Loaded eval-mode causal LM (not modified).
        tokenizer: Shared tokenizer.
        model_info: Structural metadata.
        device: Target torch.device.
        n_permutations: Number of permutation seeds to evaluate.
        permute_fraction: Fraction of layers to permute per trial.
        n_components: Number of PCA components used for directions.
        output_dir: Root output directory.
        pca_root: Root of ``results/pca_directions/`` tree.
        seed: Base random seed.
        data_root: Root of ``data/`` tree.
    """
    logger = logging.getLogger(__name__)
    pca_dir = pca_root / model_key / behavior

    # ---- Load original directions ----
    original_directions = _load_original_directions(
        model_key=model_key,
        behavior=behavior,
        pca_root=pca_root,
        model=original_model,
        tokenizer=tokenizer,
        model_info=model_info,
        device=device,
        n_components=n_components,
        seed=seed,
        data_root=data_root,
    )

    # ---- Load training pairs (must be SAME as used to fit original directions) ----
    train_pos, train_neg = _load_train_pairs(
        behavior=behavior,
        data_root=data_root,
        pca_dir=pca_dir,
        seed=seed,
    )
    logger.info(
        "Using %d training pairs for permutation invariance test.", len(train_pos)
    )

    # ---- Run permutation trials ----
    all_records: List[Dict] = []

    for perm_idx in tqdm(
        range(n_permutations),
        desc=f"{model_key}/{behavior} perms",
        unit="perm",
        dynamic_ncols=True,
        leave=False,
    ):
        # Derive a unique, reproducible seed per permutation trial
        permutation_seed = seed * 1000 + perm_idx

        layer_similarities, n_layers_permuted = _run_single_permutation(
            permutation_seed=permutation_seed,
            original_model=original_model,
            model_info=model_info,
            train_pos=train_pos,
            train_neg=train_neg,
            tokenizer=tokenizer,
            device=device,
            n_components=n_components,
            permute_fraction=permute_fraction,
            original_directions=original_directions,
        )

        for layer_idx, layer_name in enumerate(model_info.layer_module_names):
            sim = layer_similarities.get(layer_name, float("nan"))
            all_records.append(
                {
                    "permutation_seed": permutation_seed,
                    "perm_trial_idx": perm_idx,
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "subspace_cosine_sim": sim,
                    "n_layers_permuted": n_layers_permuted,
                    "model_key": model_key,
                    "behavior": behavior,
                }
            )

    # ---- Aggregate ----
    scores_df = pd.DataFrame(all_records)
    valid_sims = scores_df["subspace_cosine_sim"].dropna()
    mean_sim = float(valid_sims.mean()) if len(valid_sims) > 0 else float("nan")
    std_sim = float(valid_sims.std()) if len(valid_sims) > 1 else float("nan")
    min_sim = float(valid_sims.min()) if len(valid_sims) > 0 else float("nan")
    max_sim = float(valid_sims.max()) if len(valid_sims) > 0 else float("nan")
    claim_supported = mean_sim >= INVARIANCE_THRESHOLD if not math.isnan(mean_sim) else False

    logger.info(
        "Permutation invariance for %s/%s: mean cosine sim = %.4f ± %.4f  "
        "(min=%.4f, max=%.4f, n_perm=%d)  claim_supported=%s",
        model_key,
        behavior,
        mean_sim,
        std_sim,
        min_sim,
        max_sim,
        n_permutations,
        claim_supported,
    )

    # ---- Persist outputs ----
    artefact_dir = output_dir / model_key / behavior
    artefact_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artefact_dir / "invariance_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    logger.info("Invariance scores CSV saved → %s", csv_path)

    pt_path = artefact_dir / "invariance_scores.pt"
    torch.save(
        {
            "records": all_records,
            "model_key": model_key,
            "behavior": behavior,
            "seed": seed,
            "n_permutations": n_permutations,
            "permute_fraction": permute_fraction,
        },
        str(pt_path),
    )
    logger.info(".pt checkpoint saved → %s", pt_path)

    summary = {
        "model_key": model_key,
        "behavior": behavior,
        "mean_cosine_sim": mean_sim,
        "std_cosine_sim": std_sim,
        "min_cosine_sim": min_sim,
        "max_cosine_sim": max_sim,
        "n_permutations": n_permutations,
        "n_layers": model_info.num_layers,
        "permute_fraction": permute_fraction,
        "n_valid_scores": int(len(valid_sims)),
        "invariance_threshold": INVARIANCE_THRESHOLD,
        "claim_supported": bool(claim_supported),
        "seed": seed,
    }
    summary_path = artefact_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary JSON saved → %s", summary_path)


def _run_invariance_for_model(
    model_key: str,
    model_cfg: Dict,
    behaviors: List[str],
    device: torch.device,
    n_permutations: int,
    permute_fraction: float,
    n_components: int,
    output_dir: Path,
    pca_root: Path,
    seed: int,
    data_root: Path,
) -> None:
    """Load one model and iterate over all target behaviors for invariance testing.

    The model is loaded once and passed (not copied) to each behavior trial.
    Each permutation trial creates its own deep copy internally via
    ``apply_neuron_permutation``.

    Args:
        model_key: Short architecture key.
        model_cfg: Config sub-dict from ``models.yml``.
        behaviors: List of behavior names to test.
        device: Target torch.device.
        n_permutations: Number of permutation seeds per (model, behavior) pair.
        permute_fraction: Fraction of layers to permute per trial.
        n_components: Number of PCA components.
        output_dir: Root output directory.
        pca_root: Root of ``results/pca_directions/`` tree.
        seed: Base random seed.
        data_root: Root of ``data/`` tree.
    """
    logger = logging.getLogger(__name__)
    hf_id: str = model_cfg["huggingface_id"]

    logger.info("Loading model: %s  →  %s", hf_id, device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    original_model.to(device)
    original_model.eval()
    # Freeze all parameters — this model is strictly read-only throughout.
    for p in original_model.parameters():
        p.requires_grad_(False)

    model_info: ModelInfo = detect_model_info(original_model, hf_id)
    logger.info(
        "Model loaded. Params=%.1fB  Layers=%d  Hidden=%d  GPU=%.2f GiB",
        sum(p.numel() for p in original_model.parameters()) / 1e9,
        model_info.num_layers,
        model_info.hidden_size,
        _gpu_mem_gb(device),
    )

    for behavior in tqdm(
        behaviors, desc=f"{model_key} behaviors", unit="beh", dynamic_ncols=True
    ):
        logger.info("-" * 60)
        logger.info("Model=%s  Behavior=%s", model_key, behavior)
        logger.info("-" * 60)
        _run_invariance_for_behavior(
            model_key=model_key,
            behavior=behavior,
            original_model=original_model,
            tokenizer=tokenizer,
            model_info=model_info,
            device=device,
            n_permutations=n_permutations,
            permute_fraction=permute_fraction,
            n_components=n_components,
            output_dir=output_dir,
            pca_root=pca_root,
            seed=seed,
            data_root=data_root,
        )

    logger.info(
        "Model '%s' invariance testing complete.  GPU=%.2f GiB",
        model_key,
        _gpu_mem_gb(device),
    )
    del original_model, tokenizer
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
            "Key experiment: prove PCA behavioral directions are invariant under "
            "neuron permutations, establishing they live in weight-space "
            "symmetry-invariant subspaces.  Expected result: mean subspace "
            "cosine similarity > 0.85 across all layers and permutations."
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
        "--behavior",
        type=str,
        choices=SUPPORTED_BEHAVIORS + ["all"],
        default="all",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5,
        dest="n_permutations",
        help="Number of different random permutation seeds to evaluate.",
    )
    parser.add_argument(
        "--permute-fraction",
        type=float,
        default=0.5,
        dest="permute_fraction",
        help="Fraction of transformer layers to permute in each trial (0 < f <= 1).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        dest="n_components",
        help="Number of PCA components (must match value used in script 02).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/permutation_invariance",
        dest="output_dir",
    )
    parser.add_argument(
        "--pca-dir",
        type=str,
        default="results/pca_directions",
        dest="pca_dir",
        help="Root of pre-fitted PCA directions (output of script 02).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the permutation invariance experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ---- Fail-fast validation ----
    if not (0.0 < args.permute_fraction <= 1.0):
        parser.error(
            f"--permute-fraction must be in (0, 1], got {args.permute_fraction}."
        )
    if args.n_permutations < 1:
        parser.error(f"--n-permutations must be >= 1, got {args.n_permutations}.")
    if args.n_components < 1:
        parser.error(f"--n-components must be >= 1, got {args.n_components}.")

    output_dir = Path(args.output_dir)
    pca_root = Path(args.pca_dir)
    data_root = Path("data")

    tag = f"{args.model}_{args.behavior}"
    _setup_logging(tag, output_dir)
    logger = logging.getLogger(__name__)

    device = _resolve_device(args.device)
    _set_global_seed(args.seed)

    # ---- Load configs ----
    cfg_path = Path("config/models.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Models config not found: {cfg_path.resolve()}")
    with cfg_path.open() as fh:
        models_cfg: Dict = yaml.safe_load(fh)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root.resolve()}. "
            "Ensure data/behaviors/*.jsonl files are present."
        )

    # ---- Resolve targets ----
    all_model_keys = list(models_cfg["models"].keys())
    target_keys: List[str] = all_model_keys if args.model == "all" else [args.model]
    target_behaviors: List[str] = (
        SUPPORTED_BEHAVIORS if args.behavior == "all" else [args.behavior]
    )

    logger.info(
        "Permutation invariance experiment: "
        "models=%s  behaviors=%s  n_permutations=%d  "
        "permute_fraction=%.2f  n_components=%d  device=%s  seed=%d",
        target_keys,
        target_behaviors,
        args.n_permutations,
        args.permute_fraction,
        args.n_components,
        device,
        args.seed,
    )
    logger.info(
        "Invariance claim threshold: mean cosine sim > %.2f", INVARIANCE_THRESHOLD
    )

    for model_key in tqdm(target_keys, desc="Models", unit="model", dynamic_ncols=True):
        logger.info("=" * 72)
        logger.info("MODEL: %s", model_key)
        logger.info("=" * 72)
        _run_invariance_for_model(
            model_key=model_key,
            model_cfg=models_cfg["models"][model_key],
            behaviors=target_behaviors,
            device=device,
            n_permutations=args.n_permutations,
            permute_fraction=args.permute_fraction,
            n_components=args.n_components,
            output_dir=output_dir,
            pca_root=pca_root,
            seed=args.seed,
            data_root=data_root,
        )

    logger.info(
        "Permutation invariance experiment complete.  "
        "Processed %d model(s) × %d behavior(s) × %d permutation(s) = %d total trials.",
        len(target_keys),
        len(target_behaviors),
        args.n_permutations,
        len(target_keys) * len(target_behaviors) * args.n_permutations,
    )


if __name__ == "__main__":
    main()
