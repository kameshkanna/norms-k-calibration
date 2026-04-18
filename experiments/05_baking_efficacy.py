"""
experiments/05_baking_efficacy.py

Baking Efficacy Experiment — compares four steering methods on behavioral
alignment accuracy:

  1. none               — baseline generation (no steering)
  2. raw_addition       — mean contrastive diff added directly (no PCA)
  3. pca_uncalibrated   — PCA directions, K = 1.0 (no formula)
  4. pca_k_calibrated   — PCA directions + K = μ/√d calibration (full method)

Metric: "activation direction accuracy" — for each test pair, the steered
output activation is cosine-compared against the positive and negative
reference activations.  A prediction is correct when:

    cosine_sim(act_steered, act_positive) > cosine_sim(act_steered, act_negative)

Outputs (per model / behavior):
  results/efficacy/{model}/{behavior}/comparison.csv
  results/efficacy/{model}/{behavior}/per_pair_results.csv
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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_baking.baker import Baker
from activation_baking.calibrator import KCalibrator
from activation_baking.evaluator import BehavioralEvaluator, EvaluationResult
from activation_baking.extractor import ActivationExtractor
from activation_baking.model_utils import ModelInfo, detect_model_info
from activation_baking.pca_director import BehavioralDirections, PCADirector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("05_baking_efficacy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS: Tuple[str, ...] = ("none", "raw_addition", "pca_uncalibrated", "pca_k_calibrated")
REFERENCE_LAYER_FRACTION: float = 0.6  # use layer at ~60% depth for metric

ALL_BEHAVIORS: Tuple[str, ...] = (
    "sycophancy_suppression",
    "refusal_calibration",
    "verbosity_control",
    "formality",
    "uncertainty_expression",
)
ALL_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_behavior_data(
    data_dir: Path,
    behavior: str,
) -> Tuple[List[str], List[str]]:
    """Load positive and negative prompts for a named behavior.

    Supported formats (checked in order):
      1. JSONL file ``data/behaviors/{behavior}.jsonl`` — one JSON object per line
         with ``{"positive": "...", "negative": "..."}``.
      2. JSON file ``data/behaviors/{behavior}.json`` — top-level dict with
         ``{"positive": [...], "negative": [...]}``.
      3. Two plain text files ``{behavior}_positive.txt`` /
         ``{behavior}_negative.txt``, one prompt per line.

    Args:
        data_dir: Root data directory.
        behavior: Behavior identifier string.

    Returns:
        Tuple of (positive_prompts, negative_prompts).

    Raises:
        FileNotFoundError: If no matching data file is found.
        ValueError: If the positive/negative lists differ in length.
    """
    jsonl_path = data_dir / "behaviors" / f"{behavior}.jsonl"
    json_path = data_dir / "behaviors" / f"{behavior}.json"
    pos_path = data_dir / "behaviors" / f"{behavior}_positive.txt"
    neg_path = data_dir / "behaviors" / f"{behavior}_negative.txt"

    if jsonl_path.exists():
        positive: List[str] = []
        negative: List[str] = []
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                positive.append(record["positive"])
                negative.append(record["negative"])
    elif json_path.exists():
        with json_path.open() as fh:
            payload = json.load(fh)
        positive = payload["positive"]
        negative = payload["negative"]
    elif pos_path.exists() and neg_path.exists():
        positive = [ln.strip() for ln in pos_path.read_text().splitlines() if ln.strip()]
        negative = [ln.strip() for ln in neg_path.read_text().splitlines() if ln.strip()]
    else:
        raise FileNotFoundError(
            f"No behavior data found for '{behavior}'. "
            f"Expected {jsonl_path}, {json_path}, or {pos_path}/{neg_path}."
        )

    if len(positive) != len(negative):
        raise ValueError(
            f"Prompt list length mismatch for '{behavior}': "
            f"positive={len(positive)}, negative={len(negative)}."
        )
    return positive, negative


def load_split_indices(
    results_dir: Path,
    model_slug: str,
    behavior: str,
) -> Tuple[List[int], List[int]]:
    """Load train/test split indices saved by a prior PCA extraction run.

    Falls back to a deterministic 80/20 split when the file is absent.

    Args:
        results_dir: Root results directory.
        model_slug: Sanitised model identifier used in directory naming.
        behavior: Behavior identifier string.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    split_path = (
        results_dir / "pca_directions" / model_slug / behavior / "split_indices.json"
    )
    if split_path.exists():
        with split_path.open() as fh:
            split = json.load(fh)
        return split["train"], split["test"]

    logger.warning(
        "split_indices.json not found at %s — using default 80/20 split.", split_path
    )
    return [], []  # caller will generate indices from data length


def make_splits(
    n_total: int,
    train_indices: List[int],
    test_indices: List[int],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Return train/test index lists, generating from scratch when empty.

    Args:
        n_total: Total number of available prompt pairs.
        train_indices: Pre-computed train indices (may be empty).
        test_indices: Pre-computed test indices (may be empty).
        train_fraction: Fraction used for training when generating fresh splits.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    if train_indices and test_indices:
        # Filter to valid range in case data was trimmed.
        train_indices = [i for i in train_indices if i < n_total]
        test_indices = [i for i in test_indices if i < n_total]
        if train_indices and test_indices:
            return train_indices, test_indices

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    cut = int(n_total * train_fraction)
    return indices[:cut], indices[cut:]


# ---------------------------------------------------------------------------
# Activation extraction helpers
# ---------------------------------------------------------------------------

def _reference_layer_idx(model_info: ModelInfo) -> int:
    """Return the layer index closest to REFERENCE_LAYER_FRACTION of total depth.

    Args:
        model_info: Populated ModelInfo for the loaded model.

    Returns:
        Integer layer index.
    """
    return min(
        int(model_info.num_layers * REFERENCE_LAYER_FRACTION),
        model_info.num_layers - 1,
    )


@torch.no_grad()
def extract_last_token_activation(
    extractor: ActivationExtractor,
    prompts: List[str],
    layer_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Extract the last-token hidden state at ``layer_name`` for each prompt.

    Delegates to :py:meth:`ActivationExtractor.extract` which processes prompts
    in internal batches and returns a CPU tensor already aggregated to the last
    non-padding token position.

    Args:
        extractor: Instantiated ActivationExtractor.
        prompts: List of text prompts to process.
        layer_name: Dot-separated module path (e.g. ``"model.layers.19"``).
        device: Target device for the returned tensor.

    Returns:
        Float32 tensor of shape ``[len(prompts), hidden_size]`` on ``device``.
    """
    acts_dict = extractor.extract(prompts, layer_names=[layer_name], position="last")
    return acts_dict[layer_name].to(device)  # [N, H]


# ---------------------------------------------------------------------------
# Direction utilities
# ---------------------------------------------------------------------------

def compute_mean_diff_vector(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean contrastive difference vector (raw_addition baseline).

    Args:
        positive_acts: Activations for positive prompts, shape ``[N, H]``.
        negative_acts: Activations for negative prompts, shape ``[N, H]``.

    Returns:
        Unit-normalised mean difference vector of shape ``[H]``.
    """
    diff = (positive_acts - negative_acts).mean(dim=0)
    return F.normalize(diff.unsqueeze(0), dim=-1).squeeze(0)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_direction_accuracy(
    steered_acts: torch.Tensor,
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
) -> Tuple[float, float]:
    """Compute activation direction accuracy and mean cosine shift.

    A test pair is "correct" when the steered activation is closer to the
    positive reference than to the negative reference in cosine space.

    Args:
        steered_acts: Steered output activations, shape ``[N, H]``.
        positive_acts: Positive reference activations, shape ``[N, H]``.
        negative_acts: Negative reference activations, shape ``[N, H]``.

    Returns:
        Tuple of (accuracy, mean_cosine_shift) where both are floats in [0, 1]
        and [-1, 1] respectively.
    """
    cos_pos = F.cosine_similarity(steered_acts, positive_acts, dim=-1)  # [N]
    cos_neg = F.cosine_similarity(steered_acts, negative_acts, dim=-1)  # [N]
    correct = (cos_pos > cos_neg).float()
    accuracy = correct.mean().item()
    mean_cosine_shift = (cos_pos - cos_neg).mean().item()
    return accuracy, mean_cosine_shift


def compute_kl_divergence(
    logits_steered: torch.Tensor,
    logits_baseline: torch.Tensor,
) -> float:
    """Compute mean token-level KL divergence between steered and baseline logits.

    Args:
        logits_steered: Token logits from steered model, shape ``[N, V]``.
        logits_baseline: Token logits from baseline model, shape ``[N, V]``.

    Returns:
        Mean KL divergence as a scalar float.
    """
    log_p = F.log_softmax(logits_steered.float(), dim=-1)
    q = F.softmax(logits_baseline.float(), dim=-1)
    kl = F.kl_div(log_p, q, reduction="batchmean")
    return kl.item()


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_none(
    baker: Baker,
    test_positive: List[str],
    test_negative: List[str],
    extractor: ActivationExtractor,
    ref_layer_name: str,
    device: torch.device,
) -> Dict:
    """Evaluate the no-steering baseline.

    Args:
        baker: Fitted Baker instance (used only for generate_baseline).
        test_positive: Positive test prompts.
        test_negative: Negative test prompts.
        extractor: ActivationExtractor for hidden state extraction.
        ref_layer_name: Module path of the reference layer (e.g. "model.layers.19").
        device: Computation device.

    Returns:
        Dict with keys: method, alpha, per_pair_cos_pos, per_pair_cos_neg.
    """
    logger.info("Evaluating method: none")
    baseline_outputs = baker.generate_baseline(test_positive)

    steered_acts = extract_last_token_activation(extractor, baseline_outputs, ref_layer_name, device)
    pos_acts = extract_last_token_activation(extractor, test_positive, ref_layer_name, device)
    neg_acts = extract_last_token_activation(extractor, test_negative, ref_layer_name, device)

    cos_pos = F.cosine_similarity(steered_acts, pos_acts, dim=-1).cpu().tolist()
    cos_neg = F.cosine_similarity(steered_acts, neg_acts, dim=-1).cpu().tolist()

    return {
        "method": "none",
        "alpha": 0.0,
        "per_pair_cos_pos": cos_pos,
        "per_pair_cos_neg": cos_neg,
        "baseline_outputs": baseline_outputs,
    }


def evaluate_steered_method(
    baker: Baker,
    method_name: str,
    alpha: float,
    test_positive: List[str],
    test_negative: List[str],
    extractor: ActivationExtractor,
    ref_layer_name: str,
    device: torch.device,
) -> Dict:
    """Evaluate a steered method via Baker.generate.

    Args:
        baker: Fitted Baker instance.
        method_name: Human-readable method label for logging.
        alpha: Steering strength multiplier.
        test_positive: Positive test prompts (steered from these).
        test_negative: Negative test prompts (used for metric only).
        extractor: ActivationExtractor for hidden state extraction.
        ref_layer_name: Module path of the reference layer (e.g. "model.layers.19").
        device: Computation device.

    Returns:
        Dict with keys: method, alpha, per_pair_cos_pos, per_pair_cos_neg.
    """
    logger.info("Evaluating method: %s (alpha=%.2f)", method_name, alpha)
    steered_outputs = baker.generate(test_positive, alpha=alpha)

    steered_acts = extract_last_token_activation(extractor, steered_outputs, ref_layer_name, device)
    pos_acts = extract_last_token_activation(extractor, test_positive, ref_layer_name, device)
    neg_acts = extract_last_token_activation(extractor, test_negative, ref_layer_name, device)

    cos_pos = F.cosine_similarity(steered_acts, pos_acts, dim=-1).cpu().tolist()
    cos_neg = F.cosine_similarity(steered_acts, neg_acts, dim=-1).cpu().tolist()

    return {
        "method": method_name,
        "alpha": alpha,
        "per_pair_cos_pos": cos_pos,
        "per_pair_cos_neg": cos_neg,
        "steered_outputs": steered_outputs,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_efficacy_experiment(
    model_id: str,
    behavior: str,
    device: torch.device,
    output_dir: Path,
    data_dir: Path,
    results_dir: Path,
    alpha: float,
    seed: int,
    train_fraction: float = 0.8,
) -> None:
    """Run the full baking efficacy comparison for one model × behavior pair.

    Args:
        model_id: HuggingFace model identifier.
        behavior: Behavior name (must match a file in data/behaviors/).
        device: Torch device for computation.
        output_dir: Directory to write CSV outputs.
        data_dir: Root data directory containing behavior files.
        results_dir: Root results directory containing pca_directions/ etc.
        alpha: Steering strength for steered methods.
        seed: Random seed for reproducibility.
        train_fraction: Fraction of data used for training.

    Raises:
        FileNotFoundError: If behavior data is absent.
        RuntimeError: If model loading fails.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_slug = model_id.replace("/", "__")
    save_dir = output_dir / model_slug / behavior
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading behavior data for '%s'", behavior)
    positive_prompts, negative_prompts = load_behavior_data(data_dir, behavior)
    n_total = len(positive_prompts)

    saved_train, saved_test = load_split_indices(results_dir, model_slug, behavior)
    train_idx, test_idx = make_splits(
        n_total, saved_train, saved_test, train_fraction, seed
    )

    train_pos = [positive_prompts[i] for i in train_idx]
    train_neg = [negative_prompts[i] for i in train_idx]
    test_pos = [positive_prompts[i] for i in test_idx]
    test_neg = [negative_prompts[i] for i in test_idx]

    logger.info(
        "Split: %d train / %d test pairs", len(train_idx), len(test_idx)
    )

    # ------------------------------------------------------------------
    # Build and fit Baker (full method — PCA + K calibration)
    # ------------------------------------------------------------------
    logger.info("Initialising Baker for model '%s'", model_id)
    baker = Baker(model_id, device=device)  # type: ignore[call-arg]

    n_layers = None  # resolved after fit
    baker.fit(  # type: ignore[attr-defined]
        train_pos,
        train_neg,
        n_components=5,
        k_calibration="auto",
    )

    # ------------------------------------------------------------------
    # Set up extractor for metric computation
    # ------------------------------------------------------------------
    extractor: ActivationExtractor = baker._extractor  # type: ignore[attr-defined]
    model_info: ModelInfo = baker._model_info  # type: ignore[attr-defined]
    ref_layer_idx = _reference_layer_idx(model_info)
    ref_layer_name: str = model_info.layer_module_names[ref_layer_idx]
    logger.info(
        "Using reference layer %d / %d ('%s') for metric",
        ref_layer_idx,
        model_info.num_layers,
        ref_layer_name,
    )

    # ------------------------------------------------------------------
    # Method 1: none (baseline)
    # ------------------------------------------------------------------
    results_none = evaluate_none(
        baker, test_pos, test_neg, extractor, ref_layer_name, device
    )

    # ------------------------------------------------------------------
    # Method 2: raw_addition — L2-normalised mean diff vector, no PCA.
    # Uses Baker.fit(use_mean_diff=True) so steering runs through real generation
    # hooks, not a geometric estimate.  Directly replicates Turner et al. (2023).
    logger.info("Fitting raw_addition direction (use_mean_diff=True) on training set")
    baker.fit(
        train_pos,
        train_neg,
        n_components=1,          # ignored when use_mean_diff=True
        k_calibration=1.0,       # K=1.0: no spectral calibration
        use_mean_diff=True,
    )
    results_raw = evaluate_steered_method(
        baker, "raw_addition", alpha, test_pos, test_neg, extractor, ref_layer_name, device
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Method 3: pca_uncalibrated — PCA directions, K = 1.0
    # ------------------------------------------------------------------
    baker.fit(
        train_pos,
        train_neg,
        n_components=5,
        k_calibration=1.0,  # explicit K = 1.0 → no calibration
    )
    results_pca_uncal = evaluate_steered_method(
        baker, "pca_uncalibrated", alpha, test_pos, test_neg, extractor, ref_layer_name, device
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Method 4: pca_k_calibrated — PCA + auto K
    # ------------------------------------------------------------------
    baker.fit(
        train_pos,
        train_neg,
        n_components=5,
        k_calibration="auto",
    )
    results_pca_kcal = evaluate_steered_method(
        baker, "pca_k_calibrated", alpha, test_pos, test_neg, extractor, ref_layer_name, device
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate and persist results
    # ------------------------------------------------------------------
    all_results = [results_none, results_raw, results_pca_uncal, results_pca_kcal]

    comparison_rows: List[Dict] = []
    per_pair_rows: List[Dict] = []

    for res in all_results:
        cos_pos_arr = np.array(res["per_pair_cos_pos"])
        cos_neg_arr = np.array(res["per_pair_cos_neg"])
        correct = (cos_pos_arr > cos_neg_arr).astype(float)
        accuracy = correct.mean()
        mean_cos_shift = (cos_pos_arr - cos_neg_arr).mean()

        comparison_rows.append(
            {
                "method": res["method"],
                "alpha": res["alpha"],
                "accuracy": float(accuracy),
                "mean_cosine_shift": float(mean_cos_shift),
                "kl_divergence": float("nan"),  # requires logits — placeholder
            }
        )

        for pair_idx in range(len(cos_pos_arr)):
            per_pair_rows.append(
                {
                    "method": res["method"],
                    "alpha": res["alpha"],
                    "pair_idx": test_idx[pair_idx] if pair_idx < len(test_idx) else pair_idx,
                    "cos_pos": float(cos_pos_arr[pair_idx]),
                    "cos_neg": float(cos_neg_arr[pair_idx]),
                    "correct": int(correct[pair_idx]),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    per_pair_df = pd.DataFrame(per_pair_rows)

    comparison_path = save_dir / "comparison.csv"
    per_pair_path = save_dir / "per_pair_results.csv"

    comparison_df.to_csv(comparison_path, index=False)
    per_pair_df.to_csv(per_pair_path, index=False)

    logger.info("Saved comparison table  → %s", comparison_path)
    logger.info("Saved per-pair results  → %s", per_pair_path)
    logger.info("Results summary:\n%s", comparison_df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 05: Baking Efficacy — compare steering methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier, e.g. 'meta-llama/Llama-3.1-8B-Instruct'.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
        help="Behavior name (must match a file under data/behaviors/).",
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
        default=Path("results/efficacy"),
        help="Root directory for output CSVs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (used to find saved split indices).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength multiplier for steered methods.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of pairs used for training when no saved split exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 05.

    Supports ``--model all`` to iterate over all four target models and
    ``--behavior all`` to iterate over all five behavior datasets.
    """
    args = _parse_args()
    device = torch.device(args.device)

    model_ids: List[str] = list(ALL_MODELS) if args.model == "all" else [args.model]
    behaviors: List[str] = list(ALL_BEHAVIORS) if args.behavior == "all" else [args.behavior]

    for model_id in model_ids:
        for behavior in behaviors:
            logger.info(
                "=== Running efficacy experiment: model=%s | behavior=%s ===",
                model_id,
                behavior,
            )
            try:
                run_efficacy_experiment(
                    model_id=model_id,
                    behavior=behavior,
                    device=device,
                    output_dir=args.output_dir,
                    data_dir=args.data_dir,
                    results_dir=args.results_dir,
                    alpha=args.alpha,
                    seed=args.seed,
                    train_fraction=args.train_fraction,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Experiment failed for model=%s behavior=%s: %s",
                    model_id,
                    behavior,
                    exc,
                    exc_info=True,
                )
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
