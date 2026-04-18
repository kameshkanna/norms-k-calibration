"""
experiments/02_contrastive_extraction.py

Extract PCA behavioral directions from contrastive (positive / negative) prompt
pairs for each behavior × architecture combination.

For each (model, behavior) pair the script:
  1. Loads contrastive pairs from ``data/behaviors/{behavior}.jsonl``.
  2. Performs an 80 / 20 train / test split (indices saved for reproducibility).
  3. Computes per-layer activation differences (positive − negative) on the
     training split using ActivationExtractor.
  4. Fits a PCADirector with ``n_components`` principal components per layer.
  5. Serialises directions, per-component variance-explained stats, and the
     train/test split indices.

Outputs (per model × behavior)
-------------------------------
{output_dir}/{model}/{behavior}/directions.pt
    torch.save of Dict[layer_name, BehavioralDirections].
{output_dir}/{model}/{behavior}/variance_explained.csv
    Columns: layer_idx, layer_name, component_idx, variance_explained_ratio,
             cumulative_var_explained.
{output_dir}/{model}/{behavior}/split_indices.json
    Keys: train_indices, test_indices, n_pairs_total, random_seed.

Usage
-----
python experiments/02_contrastive_extraction.py \\
    --model llama --behavior sycophancy_suppression --n-components 5 \\
    --device cuda --seed 42
python experiments/02_contrastive_extraction.py --model all --behavior all
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
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed

from activation_baking.model_utils import ModelInfo, detect_model_info
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(tag: str, output_dir: Path) -> None:
    """Configure root logger to write to console and a timestamped file.

    Args:
        tag: Short identifier used in the log filename (e.g. ``"llama_sycophancy"``).
        output_dir: Base output directory; logs go under ``results/logs/``.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"02_contrastive_extraction_{tag}_{timestamp}.log"

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
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed applied to Python, NumPy, PyTorch, and Transformers.
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
        device_str: Requested device string such as ``"cuda"`` or ``"cpu"``.

    Returns:
        A validated torch.device.
    """
    logger = logging.getLogger(__name__)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _gpu_mem_gb(device: torch.device) -> float:
    """Return current GPU allocation in GiB, or 0.0 for non-CUDA devices.

    Args:
        device: The torch device to query.

    Returns:
        Allocated memory in GiB.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_contrastive_pairs(behavior: str, data_root: Path) -> Tuple[List[str], List[str]]:
    """Load positive and negative prompt lists from a JSONL behavior file.

    Each line of the file must be a JSON object with keys ``"positive"`` and
    ``"negative"``.

    Args:
        behavior: Behavior name matching a file in ``data/behaviors/``.
        data_root: Root path of the data directory containing ``behaviors/``.

    Returns:
        Tuple of (positives, negatives) where both lists have equal length.

    Raises:
        FileNotFoundError: If the behavior JSONL file is absent.
        ValueError: If any line is missing a required key or lists have unequal length.
    """
    jsonl_path = data_root / "behaviors" / f"{behavior}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Behavior data not found: {jsonl_path.resolve()}\n"
            f"Expected JSONL with keys 'positive' and 'negative' per line."
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

    if len(positives) != len(negatives):
        raise ValueError(
            f"Mismatched positive/negative counts: {len(positives)} vs {len(negatives)}"
        )
    if len(positives) == 0:
        raise ValueError(f"No contrastive pairs found in {jsonl_path}")

    return positives, negatives


def _train_test_split_indices(
    n: int,
    train_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Generate reproducible train / test index splits.

    Args:
        n: Total number of samples.
        train_fraction: Fraction of samples allocated to training (e.g. 0.8).
        seed: Seed for the shuffle RNG.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(np.floor(n * train_fraction))
    train_idx = indices[:n_train].tolist()
    test_idx = indices[n_train:].tolist()
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(
    hf_id: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace causal LM and its tokenizer onto the target device.

    Args:
        hf_id: HuggingFace model repository identifier.
        device: Target torch.device.
        logger: Logger for progress messages.

    Returns:
        Tuple of (model, tokenizer) with model in eval mode on ``device``.
    """
    logger.info("Loading tokenizer: %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s  →  %s", hf_id, device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    logger.info(
        "Model loaded. Params: %.1fB  GPU mem: %.2f GiB",
        sum(p.numel() for p in model.parameters()) / 1e9,
        _gpu_mem_gb(device),
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _build_variance_df(
    directions: Dict[str, BehavioralDirections],
    layer_names: List[str],
) -> pd.DataFrame:
    """Construct a tidy DataFrame of per-layer, per-component variance explained.

    Args:
        directions: Fitted behavioral directions keyed by layer name.
        layer_names: Ordered list of layer names for index alignment.

    Returns:
        DataFrame with columns: layer_idx, layer_name, component_idx,
        variance_explained_ratio, cumulative_var_explained.
    """
    records = []
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name not in directions:
            continue
        bd = directions[layer_name]
        cumsum = 0.0
        for comp_idx, ratio in enumerate(bd.explained_variance_ratio):
            cumsum += float(ratio)
            records.append(
                {
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "component_idx": comp_idx,
                    "variance_explained_ratio": float(ratio),
                    "cumulative_var_explained": cumsum,
                }
            )
    return pd.DataFrame(records)


def _run_extraction_for_behavior(
    model_key: str,
    behavior: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_info: ModelInfo,
    device: torch.device,
    n_components: int,
    output_dir: Path,
    seed: int,
    data_root: Path,
) -> None:
    """Execute contrastive extraction for one (model, behavior) pair.

    Loads data, splits train/test, extracts contrastive activation diffs,
    fits PCA, and saves all artefacts.

    Args:
        model_key: Short model key (e.g. ``"llama"``).
        behavior: Behavior name (e.g. ``"sycophancy_suppression"``).
        model: Loaded and eval-mode causal LM.
        tokenizer: Corresponding tokenizer.
        model_info: Structural metadata for this model.
        device: Target torch.device.
        n_components: Number of PCA components to retain.
        output_dir: Root output directory.
        seed: Random seed for the train/test split.
        data_root: Root of the ``data/`` directory tree.
    """
    logger = logging.getLogger(__name__)

    # ---- Load data ----
    positives, negatives = _load_contrastive_pairs(behavior, data_root)
    n_pairs = len(positives)
    logger.info(
        "Loaded %d contrastive pairs for behavior '%s'.", n_pairs, behavior
    )

    # ---- Train / test split ----
    train_idx, test_idx = _train_test_split_indices(
        n=n_pairs, train_fraction=0.8, seed=seed
    )
    train_pos = [positives[i] for i in train_idx]
    train_neg = [negatives[i] for i in train_idx]

    logger.info(
        "Split: %d train / %d test  (seed=%d).",
        len(train_idx),
        len(test_idx),
        seed,
    )

    # ---- Extraction ----
    extractor = ActivationExtractor(
        model=model, tokenizer=tokenizer, model_info=model_info, device=device
    )

    logger.info("Extracting contrastive diffs for %d train pairs…", len(train_pos))
    activation_diffs: Dict[str, torch.Tensor] = extractor.extract_contrastive_diffs(
        positive_prompts=train_pos,
        negative_prompts=train_neg,
        layer_names=model_info.layer_module_names,
    )

    # ---- Log diff norm statistics ----
    for layer_name, diffs in activation_diffs.items():
        mean_diff_norm = diffs.norm(dim=-1).mean().item()
        logger.debug(
            "  Layer %-28s  diff_norm μ=%.4f  shape=%s",
            layer_name,
            mean_diff_norm,
            tuple(diffs.shape),
        )

    # ---- Fit PCA ----
    director = PCADirector()
    directions: Dict[str, BehavioralDirections] = director.fit(
        activation_diffs=activation_diffs,
        n_components=n_components,
    )

    # ---- Log EVR per layer (top-5 components) ----
    logger.info("Explained variance ratios (top-%d components per layer):", n_components)
    for layer_name in model_info.layer_module_names:
        if layer_name not in directions:
            continue
        bd = directions[layer_name]
        evr_str = "  ".join(
            f"PC{i}={v:.4f}" for i, v in enumerate(bd.explained_variance_ratio)
        )
        logger.info("  %-30s  %s", layer_name, evr_str)

    # ---- Persist artefacts ----
    artefact_dir = output_dir / model_key / behavior
    artefact_dir.mkdir(parents=True, exist_ok=True)

    # directions.pt — torch.save the full directions dict
    pt_path = artefact_dir / "directions.pt"
    torch.save(directions, str(pt_path))
    logger.info("Directions saved → %s", pt_path)

    # raw_diffs.pt — contrastive diff matrices keyed by layer, shape [n_pairs, hidden_size].
    # Required by experiment 07 for reliable CKA computation over many samples rather
    # than over the small number of PCA components.
    raw_diffs_path = artefact_dir / "raw_diffs.pt"
    torch.save(
        {layer: diffs.cpu() for layer, diffs in activation_diffs.items()},
        str(raw_diffs_path),
    )
    logger.info("Raw contrastive diffs saved → %s", raw_diffs_path)

    # variance_explained.csv
    var_df = _build_variance_df(directions, model_info.layer_module_names)
    var_csv_path = artefact_dir / "variance_explained.csv"
    var_df.to_csv(var_csv_path, index=False)
    logger.info("Variance explained CSV saved → %s", var_csv_path)

    # split_indices.json
    split_json = {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "n_pairs_total": n_pairs,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_fraction": 0.8,
        "random_seed": seed,
        "model_key": model_key,
        "behavior": behavior,
    }
    split_path = artefact_dir / "split_indices.json"
    with split_path.open("w", encoding="utf-8") as fh:
        json.dump(split_json, fh, indent=2)
    logger.info("Split indices saved → %s", split_path)

    # ---- Explicit cleanup ----
    del extractor, activation_diffs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_extraction_for_model(
    model_key: str,
    model_cfg: Dict,
    behaviors: List[str],
    device: torch.device,
    n_components: int,
    output_dir: Path,
    seed: int,
    data_root: Path,
) -> None:
    """Load one model and iterate over all target behaviors.

    The model is loaded once and reused across all behaviors to avoid
    redundant I/O.

    Args:
        model_key: Short model key (e.g. ``"qwen"``).
        model_cfg: Config sub-dict from ``models.yml``.
        behaviors: List of behavior names to process.
        device: Target torch.device.
        n_components: Number of PCA components.
        output_dir: Root output directory.
        seed: Random seed.
        data_root: Root of ``data/`` directory tree.
    """
    logger = logging.getLogger(__name__)
    hf_id: str = model_cfg["huggingface_id"]

    t_start = time.time()
    model, tokenizer = _load_model_and_tokenizer(hf_id, device, logger)
    model_info: ModelInfo = detect_model_info(model, hf_id)

    for behavior in tqdm(behaviors, desc=f"{model_key} behaviors", unit="beh", dynamic_ncols=True):
        logger.info("-" * 60)
        logger.info("Model=%s  Behavior=%s", model_key, behavior)
        logger.info("-" * 60)
        _run_extraction_for_behavior(
            model_key=model_key,
            behavior=behavior,
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            device=device,
            n_components=n_components,
            output_dir=output_dir,
            seed=seed,
            data_root=data_root,
        )

    elapsed = time.time() - t_start
    logger.info(
        "Finished model '%s' in %.1fs  GPU mem=%.2f GiB",
        model_key,
        elapsed,
        _gpu_mem_gb(device),
    )

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for this script.

    Returns:
        Fully configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract PCA behavioral directions from contrastive prompt pairs. "
            "Outputs directions.pt, variance_explained.csv, and split_indices.json "
            "for each (model, behavior) pair."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "qwen", "gemma", "mistral", "all"],
        default="all",
        help="Which model(s) to run extraction for.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        choices=SUPPORTED_BEHAVIORS + ["all"],
        default="all",
        help="Which behavior(s) to extract directions for.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        dest="n_components",
        help="Number of PCA components to retain per layer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pca_directions",
        dest="output_dir",
        help="Root output directory for all artefacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for contrastive extraction experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ---- Fail-fast validation ----
    if args.n_components < 1:
        parser.error(f"--n-components must be >= 1, got {args.n_components}.")

    output_dir = Path(args.output_dir)
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

    data_root = Path("data")
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root.resolve()}. "
            "Ensure data/behaviors/*.jsonl files are present."
        )

    # ---- Resolve target models and behaviors ----
    all_model_keys = list(models_cfg["models"].keys())
    target_keys: List[str] = all_model_keys if args.model == "all" else [args.model]
    target_behaviors: List[str] = (
        SUPPORTED_BEHAVIORS if args.behavior == "all" else [args.behavior]
    )

    logger.info(
        "Contrastive extraction: models=%s  behaviors=%s  n_components=%d  "
        "device=%s  seed=%d",
        target_keys,
        target_behaviors,
        args.n_components,
        device,
        args.seed,
    )

    for model_key in tqdm(target_keys, desc="Models", unit="model", dynamic_ncols=True):
        logger.info("=" * 72)
        logger.info("MODEL: %s", model_key)
        logger.info("=" * 72)
        _run_extraction_for_model(
            model_key=model_key,
            model_cfg=models_cfg["models"][model_key],
            behaviors=target_behaviors,
            device=device,
            n_components=args.n_components,
            output_dir=output_dir,
            seed=args.seed,
            data_root=data_root,
        )

    logger.info(
        "All contrastive extraction complete. "
        "Processed %d model(s) × %d behavior(s).",
        len(target_keys),
        len(target_behaviors),
    )


if __name__ == "__main__":
    main()
