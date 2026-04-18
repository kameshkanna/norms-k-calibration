"""
experiments/01_norm_profiling.py

Profile mean L2 activation norms at every transformer layer for each target
architecture.  The per-layer mean norms μ_i form the empirical basis for the
K-calibration formula K_i = μ_i / √hidden_size used throughout this paper.

Outputs
-------
{output_dir}/{model_name}.csv
    Per-layer statistics: layer_idx, layer_name, mean_norm, std_norm,
    k_value, hidden_size, architecture.
{output_dir}/{model_name}_summary.json
    Aggregate statistics (global mean/std, peak layer, timing, GPU memory).

Usage
-----
python experiments/01_norm_profiling.py --model all --device cuda --seed 42
python experiments/01_norm_profiling.py --model llama --n-prompts 100
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 50 diverse calibration prompts spanning question-answering, creative writing,
# code generation, mathematics, opinions, and instructions.
CALIBRATION_PROMPTS: List[str] = [
    # Factual / question-answering (10)
    "What is the difference between supervised and unsupervised learning?",
    "Explain how the human immune system responds to a viral infection.",
    "Who was Ada Lovelace and what did she contribute to computing?",
    "What causes the Northern Lights and where are they best observed?",
    "How does HTTPS encryption protect data in transit over the internet?",
    "What are the main differences between mitosis and meiosis?",
    "Explain why the sky is blue using Rayleigh scattering.",
    "What is the role of the Federal Reserve in the U.S. economy?",
    "How does GPS determine your precise location anywhere on Earth?",
    "What is the Turing Test and what are its criticisms?",
    # Narrative / storytelling (8)
    "Write a short story about an astronaut who discovers an ancient signal.",
    "Describe a day in the life of a Victorian-era steam engineer.",
    "Tell me a story about a child who befriends a robot in a future city.",
    "Write a tense courtroom scene where the verdict is about to be read.",
    "Describe the final moments before a space shuttle launch from the crew's perspective.",
    "Write a humorous story about a wizard who keeps misplacing their wand.",
    "Tell a story about two rival scientists who accidentally swap research.",
    "Describe a medieval blacksmith crafting their greatest sword.",
    # Code generation (8)
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "Implement a binary search tree with insert, search, and delete operations in Python.",
    "Write a Rust function that safely reads a CSV file and returns a vector of records.",
    "Create a Python decorator that retries a function up to N times with exponential backoff.",
    "Write a SQL query to find the top 10 customers by total spend in the last 30 days.",
    "Implement a thread-safe LRU cache in Python using OrderedDict and threading.Lock.",
    "Write a JavaScript async function to batch API requests with rate limiting.",
    "Create a Python generator that streams chunks from a large file without loading it entirely.",
    # Mathematics (6)
    "Prove that there are infinitely many prime numbers.",
    "Explain the intuition behind Bayes' theorem and give a medical diagnosis example.",
    "Derive the formula for the sum of an arithmetic series step by step.",
    "What is the significance of Euler's identity and how is it derived?",
    "Explain the concept of a gradient in multivariable calculus.",
    "Solve the integral of x^2 * sin(x) using integration by parts.",
    # Opinions / reasoning (8)
    "What are the most compelling arguments for and against universal basic income?",
    "Should social media platforms be held legally responsible for content moderation?",
    "What are the ethical implications of using AI in criminal sentencing?",
    "Is nuclear energy a viable solution to climate change? Present both sides.",
    "Should governments mandate open-source code for all publicly-funded software?",
    "What are the trade-offs between privacy and security in digital surveillance?",
    "Discuss whether remote work has had a net positive or negative effect on productivity.",
    "What responsibilities do AI developers have toward the models they create?",
    # Instructions / how-to (10)
    "Explain step by step how to set up a secure SSH connection to a remote server.",
    "How do I train a custom object detection model using YOLOv8 on my own dataset?",
    "Provide a recipe for sourdough bread including the starter preparation process.",
    "Explain how to perform a code review effectively as a senior engineer.",
    "How do I configure Nginx as a reverse proxy for a Node.js application?",
    "Walk me through setting up a CI/CD pipeline with GitHub Actions for a Python project.",
    "How do I analyze a company's financial statements before making an investment?",
    "Explain how to calibrate a PID controller for a temperature regulation system.",
    "Describe best practices for conducting user research interviews for product design.",
    "How do I build and deploy a Docker container for a FastAPI application?",
]

assert len(CALIBRATION_PROMPTS) == 50, (
    f"Expected 50 calibration prompts, found {len(CALIBRATION_PROMPTS)}"
)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging(model_name: str, output_dir: Path) -> None:
    """Configure root logger to write to both console and a timestamped file.

    Args:
        model_name: Short model key used in the log filename.
        output_dir: Directory under which ``results/logs/`` will be created.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"01_norm_profiling_{model_name}_{timestamp}.log"

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
# GPU memory helper
# ---------------------------------------------------------------------------


def _gpu_mem_gb(device: torch.device) -> float:
    """Return current GPU memory allocation in GiB, or 0.0 for CPU.

    Args:
        device: The torch device being queried.

    Returns:
        Allocated GPU memory in GiB, 0.0 if device is CPU.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Core profiling logic
# ---------------------------------------------------------------------------


def _set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for full reproducibility.

    Args:
        seed: Integer seed value to apply across Python, NumPy, PyTorch,
              and HuggingFace Transformers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device, falling back gracefully.

    If the requested device is ``cuda`` but CUDA is unavailable the function
    falls back to ``cpu`` with a warning rather than raising.

    Args:
        device_str: Device string such as ``"cuda"``, ``"cuda:1"``, or ``"cpu"``.

    Returns:
        A validated torch.device.
    """
    logger = logging.getLogger(__name__)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but unavailable; falling back to CPU. "
            "Results will be significantly slower."
        )
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model_and_tokenizer(
    hf_id: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        hf_id: HuggingFace model repository identifier.
        device: Target torch device.
        logger: Logger instance for progress messages.

    Returns:
        Tuple of (model, tokenizer) with model moved to ``device`` in eval mode.

    Raises:
        OSError: If the model cannot be downloaded or loaded from cache.
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
        "Model loaded. Parameters: %.1fB  GPU mem: %.2f GiB",
        sum(p.numel() for p in model.parameters()) / 1e9,
        _gpu_mem_gb(device),
    )
    return model, tokenizer


def _profile_single_model(
    model_key: str,
    model_cfg: Dict,
    prompts: List[str],
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Run norm profiling for one architecture and persist results.

    For each layer the function collects the L2 norm of the last-token residual
    stream activation across all prompts, then computes:

        K_i = μ_i / √hidden_size

    Args:
        model_key: Short architecture key, e.g. ``"llama"``.
        model_cfg: Sub-dict from ``models.yml`` for this model.
        prompts: List of calibration prompt strings.
        device: Validated torch.device.
        output_dir: Root directory for CSV/JSON outputs.
        seed: Random seed (applied again locally for reproducibility).

    Returns:
        DataFrame with columns: layer_idx, layer_name, mean_norm, std_norm,
        k_value, hidden_size, architecture.
    """
    logger = logging.getLogger(__name__)
    _set_global_seed(seed)

    hf_id: str = model_cfg["huggingface_id"]
    architecture: str = model_cfg["architecture"]
    hidden_size: int = model_cfg["hidden_size"]

    t_start = time.time()

    model, tokenizer = _load_model_and_tokenizer(hf_id, device, logger)
    model_info: ModelInfo = detect_model_info(model, hf_id)

    extractor = ActivationExtractor(
        model=model, tokenizer=tokenizer, model_info=model_info, device=device
    )

    logger.info(
        "Profiling %d layers with %d prompts for architecture '%s' (hidden=%d).",
        model_info.num_layers,
        len(prompts),
        architecture,
        hidden_size,
    )

    # extract() returns Dict[layer_name, Tensor[n_prompts, hidden]]
    # compute mean and std of L2 norms per layer
    raw_acts: Dict[str, torch.Tensor] = extractor.extract(
        prompts=prompts,
        layer_names=model_info.layer_module_names,
        position="last",
    )

    # Build Dict[layer_name, {"mean": float, "std": float}]
    layer_norm_stats: Dict[str, Dict[str, float]] = {
        ln: {
            "mean": torch.linalg.vector_norm(acts, dim=-1).mean().item(),
            "std":  torch.linalg.vector_norm(acts, dim=-1).std().item(),
        }
        for ln, acts in raw_acts.items()
    }
    del raw_acts

    # Build DataFrame
    records = []
    sqrt_hidden = math.sqrt(hidden_size)

    for layer_idx, layer_name in enumerate(model_info.layer_module_names):
        stats = layer_norm_stats[layer_name]
        mean_norm: float = stats["mean"]
        std_norm: float = stats["std"]
        k_value: float = mean_norm / sqrt_hidden

        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "mean_norm": mean_norm,
                "std_norm": std_norm,
                "k_value": k_value,
                "hidden_size": hidden_size,
                "architecture": architecture,
            }
        )

        logger.debug(
            "  Layer %02d %-30s  μ=%.4f  σ=%.4f  K=%.6f",
            layer_idx,
            layer_name,
            mean_norm,
            std_norm,
            k_value,
        )

    df = pd.DataFrame(records)

    # ---- Persist CSV ----
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_key}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved → %s", csv_path)

    # ---- Persist .pt ----
    pt_path = output_dir / f"{model_key}.pt"
    torch.save(
        {
            "layer_norm_stats": layer_norm_stats,
            "df_records": records,
            "model_key": model_key,
            "hf_id": hf_id,
            "architecture": architecture,
            "hidden_size": hidden_size,
            "seed": seed,
            "n_prompts": len(prompts),
        },
        str(pt_path),
    )
    logger.info(".pt checkpoint saved → %s", pt_path)

    # ---- Persist summary JSON ----
    elapsed = time.time() - t_start
    gpu_mem = _gpu_mem_gb(device)

    summary = {
        "model_key": model_key,
        "hf_id": hf_id,
        "architecture": architecture,
        "hidden_size": hidden_size,
        "num_layers": model_info.num_layers,
        "n_prompts": len(prompts),
        "seed": seed,
        "global_mean_norm": float(df["mean_norm"].mean()),
        "global_std_norm": float(df["mean_norm"].std()),
        "min_mean_norm": float(df["mean_norm"].min()),
        "max_mean_norm": float(df["mean_norm"].max()),
        "peak_layer_idx": int(df["mean_norm"].idxmax()),
        "peak_layer_name": df.loc[df["mean_norm"].idxmax(), "layer_name"],
        "global_mean_k": float(df["k_value"].mean()),
        "global_std_k": float(df["k_value"].std()),
        "elapsed_seconds": round(elapsed, 2),
        "gpu_mem_gb": round(gpu_mem, 3),
    }
    json_path = output_dir / f"{model_key}_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary JSON saved → %s", json_path)

    logger.info(
        "Finished %s in %.1fs | global μ_norm=%.4f | global K=%.6f | GPU=%.2f GiB",
        model_key,
        elapsed,
        summary["global_mean_norm"],
        summary["global_mean_k"],
        gpu_mem,
    )

    # ---- Clean up ----
    del model, tokenizer, extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return df


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for this script.

    Returns:
        Fully configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Profile mean L2 activation norms at every transformer layer for "
            "each target architecture.  Outputs per-layer CSVs and summary JSONs "
            "used as input to the K-calibration validation experiment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama", "qwen", "gemma", "mistral", "all"],
        default="all",
        help="Which model(s) to profile.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (e.g. 'cuda', 'cuda:1', 'cpu').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/norm_profiles",
        dest="output_dir",
        help="Root directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        dest="n_prompts",
        help="Number of calibration prompts to use (max 50).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for norm profiling experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ---- Validation ----
    if not (1 <= args.n_prompts <= len(CALIBRATION_PROMPTS)):
        parser.error(
            f"--n-prompts must be between 1 and {len(CALIBRATION_PROMPTS)}, "
            f"got {args.n_prompts}."
        )

    output_dir = Path(args.output_dir)
    _setup_logging("all" if args.model == "all" else args.model, output_dir)
    logger = logging.getLogger(__name__)

    device = _resolve_device(args.device)
    _set_global_seed(args.seed)

    # ---- Load configs ----
    cfg_path = Path("config/models.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Models config not found: {cfg_path.resolve()}")
    with cfg_path.open() as fh:
        models_cfg: Dict = yaml.safe_load(fh)

    exp_cfg_path = Path("config/experiments.yml")
    if not exp_cfg_path.exists():
        raise FileNotFoundError(f"Experiments config not found: {exp_cfg_path.resolve()}")
    with exp_cfg_path.open() as fh:
        exp_cfg: Dict = yaml.safe_load(fh)

    # ---- Determine target models ----
    all_model_keys = list(models_cfg["models"].keys())
    if args.model == "all":
        target_keys = all_model_keys
    else:
        if args.model not in all_model_keys:
            raise ValueError(
                f"Model key '{args.model}' not found in config. "
                f"Available: {all_model_keys}"
            )
        target_keys = [args.model]

    prompts = CALIBRATION_PROMPTS[: args.n_prompts]
    logger.info(
        "Starting norm profiling: models=%s  device=%s  prompts=%d  seed=%d",
        target_keys,
        device,
        len(prompts),
        args.seed,
    )

    # ---- Run profiling per model ----
    results: Dict[str, pd.DataFrame] = {}
    for model_key in tqdm(target_keys, desc="Models", unit="model", dynamic_ncols=True):
        logger.info("=" * 72)
        logger.info("Processing model: %s", model_key)
        logger.info("=" * 72)
        model_cfg = models_cfg["models"][model_key]
        df = _profile_single_model(
            model_key=model_key,
            model_cfg=model_cfg,
            prompts=prompts,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
        )
        results[model_key] = df

    # ---- Cross-model summary ----
    if len(results) > 1:
        combined_path = output_dir / "all_models_combined.csv"
        combined_df = pd.concat(list(results.values()), ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        logger.info("Combined CSV saved → %s", combined_path)

    logger.info("Norm profiling complete for models: %s", target_keys)


if __name__ == "__main__":
    main()
