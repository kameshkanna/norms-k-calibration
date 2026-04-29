"""
experiments/08_raw_activation_control.py

Raw-Activation PC1 Control Experiment — critical specificity check for the
weight-space alignment claim in §5.1 / §6.

The paper claims that *behavioral* contrastive PCA directions align with top
singular vectors of MLP W_down at 3.69× above random.  A reviewer correctly
points out that this result is only scientifically meaningful if generic
variance-capturing directions (PC1 of *raw* activations) do NOT show the same
alignment.  This experiment runs the explicit comparison.

Five conditions are evaluated per layer per model:

  1. ``behavioral_pc1``       — PC1 of contrastive diffs h(pos) − h(neg); the
                                 paper's primary quantity.
  2. ``contrastive_mean_dir`` — Mean of contrastive diffs, normalised; probes
                                 whether PCA adds anything beyond mean-shift.
  3. ``raw_pc1``              — PC1 of raw h(x) from 50 generic calibration
                                 prompts; the critical null-hypothesis control.
  4. ``raw_mean_dir``         — Mean of raw h(x), normalised; a second generic
                                 baseline.
  5. ``random_baseline``      — Mean-max alignment of random unit vectors;
                                 the denominator shared by all ratios.

All five conditions use identical W_down left-singular-vector sets and the same
random generator seed so alignment ratios are directly comparable.

Outputs
-------
results/raw_activation_control/{model_key}/
    summary.csv
        Per-layer aggregated ratios (behavioral mean over 5 behaviors; raw_pc1;
        raw_mean_dir; random_baseline). Columns:
            layer_idx, layer_name,
            behavioral_pc1_alignment_mean, behavioral_pc1_ratio_mean,
            raw_pc1_alignment, raw_pc1_ratio,
            raw_mean_dir_alignment, raw_mean_dir_ratio,
            random_baseline
    {behavior}/
        per_layer_comparison.csv
            Columns: layer_idx, layer_name,
                     behavioral_pc1_alignment, behavioral_pc1_ratio,
                     contrastive_mean_alignment, contrastive_mean_ratio,
                     raw_pc1_alignment, raw_pc1_ratio,
                     raw_mean_dir_alignment, raw_mean_dir_ratio,
                     random_baseline

results/raw_activation_control/aggregate_summary.csv
    Cross-model table with mean ratios per condition — ready for paper Table 3.
    Columns: model_key, behavioral_pc1_ratio, raw_pc1_ratio,
             raw_mean_dir_ratio, contrastive_mean_ratio, random_baseline

Usage
-----
python experiments/08_raw_activation_control.py --model llama --device cuda
python experiments/08_raw_activation_control.py --model all --device cuda
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed

from activation_baking.model_utils import ModelInfo, detect_model_info
from activation_baking.extractor import ActivationExtractor

# ---------------------------------------------------------------------------
# Constants (re-exported from Exp 01 for reproducibility)
# ---------------------------------------------------------------------------

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

ALL_BEHAVIORS: Tuple[str, ...] = (
    "sycophancy_suppression",
    "refusal_calibration",
    "verbosity_control",
    "formality",
    "uncertainty_expression",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("08_raw_activation_control")


# ---------------------------------------------------------------------------
# Seed & device helpers
# ---------------------------------------------------------------------------


def _set_global_seed(seed: int) -> None:
    """Apply seed to Python, NumPy, PyTorch, and HuggingFace Transformers.

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
    """Resolve device string with graceful CUDA fallback.

    Args:
        device_str: E.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``.

    Returns:
        Validated torch.device.
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _gpu_mem_gb(device: torch.device) -> float:
    """Return current GPU allocation in GiB, or 0.0 on CPU.

    Args:
        device: Torch device to query.

    Returns:
        Allocated memory in GiB.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_contrastive_pairs(
    behavior: str,
    data_root: Path,
) -> Tuple[List[str], List[str]]:
    """Load positive and negative prompt lists from a JSONL behavior file.

    Args:
        behavior: Behavior name matching a file under ``data/behaviors/``.
        data_root: Root of the ``data/`` directory.

    Returns:
        Tuple of (positives, negatives) with equal-length lists.

    Raises:
        FileNotFoundError: If the behavior file is absent.
        ValueError: If any record is missing required keys.
    """
    jsonl_path = data_root / "behaviors" / f"{behavior}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Behavior JSONL not found: {jsonl_path.resolve()}"
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
                raise ValueError(f"{jsonl_path}:{line_no}: malformed JSON — {exc}") from exc
            if "positive" not in record or "negative" not in record:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: missing 'positive' or 'negative' key."
                )
            positives.append(str(record["positive"]))
            negatives.append(str(record["negative"]))

    if not positives:
        raise ValueError(f"No contrastive pairs found in {jsonl_path}")
    return positives, negatives


# ---------------------------------------------------------------------------
# Alignment computation helpers (mirrors Exp 06)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_alignment_matrix(
    direction: torch.Tensor,
    reference_vectors: torch.Tensor,
) -> torch.Tensor:
    """Absolute cosine similarity between a single direction and k reference vectors.

    Args:
        direction: Unit-length float32 tensor of shape ``[H]``.
        reference_vectors: Float32 tensor of shape ``[k, H]``.

    Returns:
        Float32 tensor of shape ``[k]`` — absolute cosine similarities.
    """
    d_n = F.normalize(direction.float().unsqueeze(0), dim=-1)   # [1, H]
    r_n = F.normalize(reference_vectors.float(), dim=-1)         # [k, H]
    return (d_n @ r_n.T).abs().squeeze(0)                        # [k]


def _mean_max_alignment_single(
    direction: torch.Tensor,
    reference_vectors: torch.Tensor,
) -> float:
    """Max absolute cosine similarity between one direction and k reference vectors.

    Because there is only one direction, mean-max reduces to the plain max.

    Args:
        direction: Float32 tensor of shape ``[H]``.
        reference_vectors: Float32 tensor of shape ``[k, H]``.

    Returns:
        Scalar float in ``[0, 1]``.
    """
    sims = _compute_alignment_matrix(direction, reference_vectors)
    return sims.max().item()


@torch.no_grad()
def _random_unit_vectors(
    k: int,
    dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample k random unit vectors in R^{dim}.

    Args:
        k: Number of vectors.
        dim: Dimensionality.
        generator: Seeded torch.Generator for reproducibility.

    Returns:
        Float32 tensor of shape ``[k, dim]``.
    """
    raw = torch.randn(k, dim, generator=generator)
    return F.normalize(raw, dim=-1)


# ---------------------------------------------------------------------------
# Weight matrix helpers (mirrors Exp 06)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _get_left_singular_vectors(
    model: AutoModelForCausalLM,
    layer_idx: int,
    model_info: ModelInfo,
    k: int,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Fetch the top-k left singular vectors of W_down for one layer.

    W_down has shape ``[hidden, intermediate]``.  Left singular vectors live in
    R^{hidden} — the same space as residual-stream PCA directions.

    Args:
        model: Loaded HuggingFace causal LM (CPU or GPU).
        layer_idx: 0-based transformer layer index.
        model_info: Structural metadata for this model.
        k: Number of top singular vectors to extract.

    Returns:
        Tuple of ``(U_top_k, hidden_size)`` where ``U_top_k`` is float32
        ``[k, hidden]``, or ``None`` if extraction fails.
    """
    proj_name = model_info.mlp_down_proj_names[layer_idx]
    parts = proj_name.split(".")
    module = model
    try:
        for part in parts:
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        W: torch.Tensor = module.weight.detach().cpu().float()  # type: ignore[attr-defined]
    except (AttributeError, IndexError) as exc:
        logger.warning("Layer %d — cannot load W_down (%s): %s", layer_idx, proj_name, exc)
        return None

    # W: [hidden, intermediate].  Economy SVD → U: [hidden, min_rank].
    actual_k = min(k, min(W.shape))
    try:
        U, _, _ = torch.linalg.svd(W, full_matrices=False)
    except torch.linalg.LinAlgError as exc:
        logger.warning("Layer %d — SVD failed: %s", layer_idx, exc)
        del W
        gc.collect()
        return None

    U_top = U[:, :actual_k].T  # [k, hidden] — each row is a left singular vector
    hidden = W.shape[0]
    del W, U
    return F.normalize(U_top.float(), dim=-1), hidden


# ---------------------------------------------------------------------------
# Direction extraction helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _fit_pc1(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Fit PCA to a data matrix and return the first principal component.

    Args:
        matrix: Float32 tensor of shape ``[n_samples, hidden]``.  Rows are
                observations; columns are features.

    Returns:
        Unit-norm float32 tensor of shape ``[hidden]``.
    """
    X = matrix.float().cpu().numpy()
    pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
    pca.fit(X)
    pc1 = torch.from_numpy(pca.components_[0].copy()).float()
    return F.normalize(pc1, dim=-1)


@torch.no_grad()
def _compute_mean_dir(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute the L2-normalised mean direction of a set of activation vectors.

    Args:
        matrix: Float32 tensor of shape ``[n_samples, hidden]``.

    Returns:
        Unit-norm float32 tensor of shape ``[hidden]``.
    """
    mean_vec = matrix.float().mean(dim=0)
    return F.normalize(mean_vec, dim=-1)


# ---------------------------------------------------------------------------
# Per-layer alignment record builder
# ---------------------------------------------------------------------------


def _compute_layer_record(
    layer_idx: int,
    layer_name: str,
    behavioral_pc1: Optional[torch.Tensor],
    contrastive_mean_dir: Optional[torch.Tensor],
    raw_pc1: torch.Tensor,
    raw_mean_dir: torch.Tensor,
    svd_vectors: torch.Tensor,   # [k, hidden]
    generator: torch.Generator,
) -> Dict:
    """Compute all alignment scores for one layer and return a record dict.

    All alignment values are mean-max over the k SVD vectors (which for a single
    direction reduces to the plain max absolute cosine similarity).

    Args:
        layer_idx: 0-based layer index.
        layer_name: Human-readable layer module name.
        behavioral_pc1: Unit-norm tensor ``[H]`` or None if unavailable.
        contrastive_mean_dir: Unit-norm tensor ``[H]`` or None if unavailable.
        raw_pc1: Unit-norm tensor ``[H]``.
        raw_mean_dir: Unit-norm tensor ``[H]``.
        svd_vectors: Float32 ``[k, H]`` left singular vectors of W_down.
        generator: Seeded generator for random baseline (re-seeded per layer to
                   ensure the same baseline is used across all conditions).

    Returns:
        Dict with alignment scores and ratios for all five conditions.
    """
    k, H = svd_vectors.shape
    # Vectorised random baseline: 200 random unit directions × k SVD vectors.
    # For each random direction, max absolute cosine similarity over k SVD vectors;
    # then average over the 200 directions to get a stable E[max_j |cos(rand, v_j)|].
    n_rand = 200
    rand_dirs = F.normalize(
        torch.randn(n_rand, H, generator=generator), dim=-1
    )  # [n_rand, H]
    rand_sims = (rand_dirs @ svd_vectors.T).abs()               # [n_rand, k]
    random_baseline: float = rand_sims.max(dim=-1).values.mean().item()

    svd_cpu = svd_vectors.cpu()

    def _ratio(alignment: float) -> float:
        return alignment / (random_baseline + 1e-9)

    raw_pc1_align = _mean_max_alignment_single(raw_pc1, svd_cpu)
    raw_mean_align = _mean_max_alignment_single(raw_mean_dir, svd_cpu)

    record: Dict = {
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "raw_pc1_alignment": raw_pc1_align,
        "raw_pc1_ratio": _ratio(raw_pc1_align),
        "raw_mean_dir_alignment": raw_mean_align,
        "raw_mean_dir_ratio": _ratio(raw_mean_align),
        "random_baseline": random_baseline,
    }

    if behavioral_pc1 is not None:
        b_align = _mean_max_alignment_single(behavioral_pc1, svd_cpu)
        record["behavioral_pc1_alignment"] = b_align
        record["behavioral_pc1_ratio"] = _ratio(b_align)
    else:
        record["behavioral_pc1_alignment"] = float("nan")
        record["behavioral_pc1_ratio"] = float("nan")

    if contrastive_mean_dir is not None:
        c_align = _mean_max_alignment_single(contrastive_mean_dir, svd_cpu)
        record["contrastive_mean_alignment"] = c_align
        record["contrastive_mean_ratio"] = _ratio(c_align)
    else:
        record["contrastive_mean_alignment"] = float("nan")
        record["contrastive_mean_ratio"] = float("nan")

    return record


# ---------------------------------------------------------------------------
# Core per-model experiment runner
# ---------------------------------------------------------------------------


def run_control_experiment(
    model_key: str,
    model_cfg: Dict,
    behaviors: List[str],
    device: torch.device,
    output_dir: Path,
    data_root: Path,
    top_k: int,
    seed: int,
) -> Optional[pd.DataFrame]:
    """Run the full raw-activation control experiment for one model.

    Loads the model once, extracts raw activations (CALIBRATION_PROMPTS) and
    contrastive diffs (all behaviors), then measures and records alignment ratios
    for all five conditions against W_down left singular vectors.

    Args:
        model_key: Short model key, e.g. ``"llama"``.
        model_cfg: Config sub-dict from ``models.yml``.
        behaviors: List of behavior names to compare against.
        device: Torch device for activation extraction.
        output_dir: Root directory for output artefacts.
        data_root: Root of ``data/`` directory tree.
        top_k: Number of top singular vectors for alignment comparison.
        seed: Global random seed.

    Returns:
        DataFrame of per-layer summary statistics across all behaviors, or
        None if the model cannot be loaded.
    """
    _set_global_seed(seed)
    hf_id: str = model_cfg["huggingface_id"]
    logger.info("=" * 72)
    logger.info("Running Exp 08 for model: %s (%s)", model_key, hf_id)
    logger.info("=" * 72)

    t_start = time.time()

    # ---- Load model ----
    logger.info("Loading tokenizer: %s", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("Loading model: %s → %s (dtype=%s)", hf_id, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_info: ModelInfo = detect_model_info(model, hf_id)
    logger.info(
        "Model loaded: %d layers, hidden=%d  GPU=%.2f GiB",
        model_info.num_layers,
        model_info.hidden_size,
        _gpu_mem_gb(device),
    )

    extractor = ActivationExtractor(
        model=model, tokenizer=tokenizer, model_info=model_info, device=device
    )
    layer_names: List[str] = model_info.layer_module_names

    # ---- Extract raw activations (once, shared across all conditions) ----
    logger.info(
        "Extracting raw activations from %d calibration prompts across %d layers…",
        len(CALIBRATION_PROMPTS),
        len(layer_names),
    )
    raw_acts: Dict[str, torch.Tensor] = extractor.extract(
        prompts=CALIBRATION_PROMPTS,
        layer_names=layer_names,
        position="last",
    )  # Dict[layer_name, Tensor[n_prompts, hidden]]
    logger.info("Raw activations extracted. GPU=%.2f GiB", _gpu_mem_gb(device))

    # Pre-compute raw PC1 and mean direction per layer
    logger.info("Fitting PC1 and mean direction from raw activations…")
    raw_pc1_by_layer: Dict[str, torch.Tensor] = {}
    raw_mean_by_layer: Dict[str, torch.Tensor] = {}
    for ln in tqdm(layer_names, desc="Raw PC1", dynamic_ncols=True, leave=False):
        acts = raw_acts[ln].cpu()   # [n_prompts, hidden]
        raw_pc1_by_layer[ln] = _fit_pc1(acts)
        raw_mean_by_layer[ln] = _compute_mean_dir(acts)

    del raw_acts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Extract contrastive diffs per behavior ----
    behavioral_pc1_by_behavior_layer: Dict[str, Dict[str, torch.Tensor]] = {}
    contrastive_mean_by_behavior_layer: Dict[str, Dict[str, torch.Tensor]] = {}

    for behavior in tqdm(behaviors, desc="Behaviors", dynamic_ncols=True):
        logger.info("Extracting contrastive diffs for behavior '%s'…", behavior)
        positives, negatives = _load_contrastive_pairs(behavior, data_root)

        contrastive_diffs: Dict[str, torch.Tensor] = extractor.extract_contrastive_diffs(
            positive_prompts=positives,
            negative_prompts=negatives,
            layer_names=layer_names,
        )  # Dict[layer_name, Tensor[n_pairs, hidden]]

        beh_pc1: Dict[str, torch.Tensor] = {}
        beh_mean: Dict[str, torch.Tensor] = {}
        for ln, diffs in contrastive_diffs.items():
            diffs_cpu = diffs.cpu()
            beh_pc1[ln] = _fit_pc1(diffs_cpu)
            beh_mean[ln] = _compute_mean_dir(diffs_cpu)

        behavioral_pc1_by_behavior_layer[behavior] = beh_pc1
        contrastive_mean_by_behavior_layer[behavior] = beh_mean

        del contrastive_diffs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Free model from device; weights needed CPU-side for SVD ----
    del extractor
    if device.type == "cuda":
        model_cpu = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model_cpu.eval()
        model_info_cpu = detect_model_info(model_cpu, hf_id)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model_cpu = model
        model_info_cpu = model_info

    # ---- Per-layer alignment computation ----
    logger.info("Computing alignment ratios across %d layers…", len(layer_names))

    # Seeded generator — same sequence used for random baseline across all layers
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Collect per-behavior records
    all_behavior_dfs: List[pd.DataFrame] = []

    for layer_idx, layer_name in tqdm(
        enumerate(layer_names),
        total=len(layer_names),
        desc="Layers",
        dynamic_ncols=True,
    ):
        svd_result = _get_left_singular_vectors(model_cpu, layer_idx, model_info_cpu, top_k)
        if svd_result is None:
            logger.warning("Layer %d: SVD failed, skipping.", layer_idx)
            continue
        svd_vectors, _ = svd_result

        # Per-behavior records
        for behavior in behaviors:
            beh_pc1 = behavioral_pc1_by_behavior_layer.get(behavior, {}).get(layer_name)
            beh_mean = contrastive_mean_by_behavior_layer.get(behavior, {}).get(layer_name)
            raw_pc1 = raw_pc1_by_layer[layer_name]
            raw_mean = raw_mean_by_layer[layer_name]

            record = _compute_layer_record(
                layer_idx=layer_idx,
                layer_name=layer_name,
                behavioral_pc1=beh_pc1,
                contrastive_mean_dir=beh_mean,
                raw_pc1=raw_pc1,
                raw_mean_dir=raw_mean,
                svd_vectors=svd_vectors,
                generator=generator,
            )
            record["behavior"] = behavior
            all_behavior_dfs.append(pd.DataFrame([record]))

        del svd_vectors
        gc.collect()

    # ---- Persist per-behavior CSVs ----
    model_out_dir = output_dir / model_key
    model_out_dir.mkdir(parents=True, exist_ok=True)

    if not all_behavior_dfs:
        logger.error("No alignment records produced for model %s — aborting save.", model_key)
        return None

    all_df = pd.concat(all_behavior_dfs, ignore_index=True)

    for behavior in behaviors:
        beh_df = all_df[all_df["behavior"] == behavior].drop(columns=["behavior"])
        beh_out = model_out_dir / behavior
        beh_out.mkdir(parents=True, exist_ok=True)
        beh_csv = beh_out / "per_layer_comparison.csv"
        beh_df.to_csv(beh_csv, index=False)
        logger.info("Saved per-layer comparison → %s", beh_csv)

        if not beh_df.empty:
            logger.info(
                "  %s — behavioral_pc1_ratio μ=%.3f | raw_pc1_ratio μ=%.3f | "
                "raw_mean_ratio μ=%.3f | random_baseline μ=%.4f",
                behavior,
                beh_df["behavioral_pc1_ratio"].mean(),
                beh_df["raw_pc1_ratio"].mean(),
                beh_df["raw_mean_dir_ratio"].mean(),
                beh_df["random_baseline"].mean(),
            )

    # ---- Build per-layer summary (mean behavioral ratio averaged over behaviors) ----
    summary_cols = [
        "layer_idx", "layer_name",
        "behavioral_pc1_alignment", "behavioral_pc1_ratio",
        "contrastive_mean_alignment", "contrastive_mean_ratio",
        "raw_pc1_alignment", "raw_pc1_ratio",
        "raw_mean_dir_alignment", "raw_mean_dir_ratio",
        "random_baseline",
    ]
    agg_fns = {c: "mean" for c in summary_cols if c not in ("layer_idx", "layer_name", "behavior")}
    summary_df = (
        all_df.groupby(["layer_idx", "layer_name"])
        .agg(agg_fns)
        .reset_index()
    )

    summary_df.columns = [
        c if c in ("layer_idx", "layer_name") else c
        for c in summary_df.columns
    ]

    # Rename aggregated behavioral columns to reflect that they are means
    rename_map = {
        "behavioral_pc1_alignment": "behavioral_pc1_alignment_mean",
        "behavioral_pc1_ratio": "behavioral_pc1_ratio_mean",
        "contrastive_mean_alignment": "contrastive_mean_alignment_mean",
        "contrastive_mean_ratio": "contrastive_mean_ratio_mean",
    }
    summary_df = summary_df.rename(columns=rename_map)

    summary_csv = model_out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved per-layer summary → %s", summary_csv)

    elapsed = time.time() - t_start
    logger.info(
        "Model %s complete in %.1fs | mean behavioral ratio=%.3f | mean raw_pc1 ratio=%.3f",
        model_key,
        elapsed,
        summary_df["behavioral_pc1_ratio_mean"].mean(),
        summary_df["raw_pc1_ratio"].mean(),
    )

    # ---- Cleanup ----
    del model_cpu, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_df


# ---------------------------------------------------------------------------
# Aggregate cross-model summary
# ---------------------------------------------------------------------------


def _build_aggregate_summary(
    per_model_summaries: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Build and save a cross-model aggregate summary CSV.

    Args:
        per_model_summaries: Dict mapping model_key → per-layer summary DataFrame.
        output_dir: Root output directory for ``aggregate_summary.csv``.
    """
    records = []
    for model_key, df in per_model_summaries.items():
        if df is None or df.empty:
            continue
        records.append(
            {
                "model_key": model_key,
                "behavioral_pc1_ratio": df["behavioral_pc1_ratio_mean"].mean(),
                "contrastive_mean_ratio": df["contrastive_mean_ratio_mean"].mean(),
                "raw_pc1_ratio": df["raw_pc1_ratio"].mean(),
                "raw_mean_dir_ratio": df["raw_mean_dir_ratio"].mean(),
                "random_baseline": df["random_baseline"].mean(),
                "behavioral_advantage_over_raw_pc1": (
                    df["behavioral_pc1_ratio_mean"].mean() - df["raw_pc1_ratio"].mean()
                ),
            }
        )

    agg_df = pd.DataFrame(records)
    csv_path = output_dir / "aggregate_summary.csv"
    agg_df.to_csv(csv_path, index=False)
    logger.info("Aggregate summary saved → %s", csv_path)

    if not agg_df.empty:
        logger.info("\n%s", agg_df.to_string(index=False))
        logger.info(
            "\nOverall mean behavioral_pc1_ratio=%.3f  raw_pc1_ratio=%.3f  "
            "advantage=%.3f",
            agg_df["behavioral_pc1_ratio"].mean(),
            agg_df["raw_pc1_ratio"].mean(),
            agg_df["behavioral_advantage_over_raw_pc1"].mean(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 08: Raw-Activation PC1 Control — verify that behavioral "
            "PCA directions align with W_down singular vectors more than generic "
            "variance-capturing directions from raw activations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model key (llama/qwen/mistral/gemma) or 'all'.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        default="all",
        help="Behavior name or 'all'.",
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
        default=Path("results/raw_activation_control"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        dest="data_root",
        help="Root data directory containing behaviors/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        help="Number of top W_down singular vectors for alignment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_MODEL_KEYS: Tuple[str, ...] = ("llama", "qwen", "mistral", "gemma")


def main() -> None:
    """Main entry point for Experiment 08."""
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_global_seed(args.seed)

    cfg_path = Path("config/models.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Models config not found: {cfg_path.resolve()}")
    with cfg_path.open() as fh:
        models_cfg: Dict = yaml.safe_load(fh)

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {args.data_root.resolve()}. "
            "Ensure data/behaviors/*.jsonl files are present."
        )

    target_model_keys: List[str] = (
        list(_MODEL_KEYS) if args.model == "all" else [args.model]
    )
    target_behaviors: List[str] = (
        list(ALL_BEHAVIORS) if args.behavior == "all" else [args.behavior]
    )

    logger.info(
        "Experiment 08 | models=%s | behaviors=%s | top_k=%d | device=%s | seed=%d",
        target_model_keys,
        target_behaviors,
        args.top_k,
        device,
        args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_model_summaries: Dict[str, Optional[pd.DataFrame]] = {}

    for model_key in tqdm(target_model_keys, desc="Models", unit="model", dynamic_ncols=True):
        if model_key not in models_cfg.get("models", {}):
            logger.error("Model key '%s' not found in config — skipping.", model_key)
            continue
        try:
            summary_df = run_control_experiment(
                model_key=model_key,
                model_cfg=models_cfg["models"][model_key],
                behaviors=target_behaviors,
                device=device,
                output_dir=args.output_dir,
                data_root=args.data_root,
                top_k=args.top_k,
                seed=args.seed,
            )
            per_model_summaries[model_key] = summary_df
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Experiment 08 failed for model=%s: %s", model_key, exc, exc_info=True
            )
            per_model_summaries[model_key] = None
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _build_aggregate_summary(per_model_summaries, args.output_dir)
    logger.info("Experiment 08 complete.")


if __name__ == "__main__":
    main()
