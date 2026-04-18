# Norm-Calibrated K for Activation Steering: A Rank-1 Weight Perturbation Derivation

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**ICML 2026 Workshop on Weight-Space Symmetries**

---

## Abstract

Activation steering is a powerful interpretability and alignment technique for shaping large language model behaviour by adding direction vectors to residual stream activations at inference time. A critical but largely unaddressed problem is the choice of steering strength K: existing approaches rely on manual grid search or heuristics that do not transfer across architectures, layers, or behaviours. We derive the first principled closed-form formula for K from a rank-1 weight perturbation equivalence: a rank-1 update delta_W = alpha * u * v^T with ||u|| = ||v|| = 1 produces an expected output shift of alpha * sqrt(hidden_size) on a unit-norm input, establishing a direct link between activation norm scale and weight-space geometry. Setting K = mean_norm / sqrt(hidden_size) normalises the steering vector magnitude to match this scale, yielding calibrated interventions that are architecture-agnostic and require zero hyperparameter tuning. We validate this formula empirically across four architectures (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B), five behavioural dimensions, and over 1,200 contrastive prompt pairs.

---

## Method

### The K Calibration Formula

The core contribution is a closed-form formula for activation steering strength, derived from first principles in weight space.

**Rank-1 Perturbation Equivalence.** Consider a rank-1 weight update applied to any linear layer in a transformer:

```
delta_W = alpha * u * v^T
```

where `u` and `v` are unit-norm vectors (||u|| = ||v|| = 1) and `alpha` is a scalar magnitude. For a unit-norm input `x` (||x|| = 1), the expected output perturbation magnitude is:

```
E[||delta_W x||] = alpha * |u^T x| * ||v|| ≈ alpha / sqrt(hidden_size)
```

by the central limit theorem applied to the inner product of two random unit vectors in R^d. Inverting this, if we want an activation-space intervention of magnitude matching a rank-1 weight perturbation with alpha equal to the actual per-layer activation norm, we set:

```
K = mean_norm / sqrt(hidden_size)
```

where `mean_norm` is the mean L2 norm of residual stream activations at the target layer, estimated over a small representative prompt set. This single scalar calibrates the steering vector magnitude in units commensurate with the effective weight-space scale at each layer.

**Why this matters geometrically.** PCA directions extracted from contrastive activations lie in the same subspace as the top singular vectors of the MLP weight matrices. Our formula anchors the steering magnitude to the operator norm scale of those projections, ensuring that the intervention energy matches the signal energy that the network itself uses to encode behavioural variation.

**Implementation:**

```python
K = mean_norm / math.sqrt(hidden_size)
```

This requires only a single forward pass over a small set of calibration prompts and no hyperparameter search. It is computed once per layer per behaviour and stored in the adapter artefact alongside the PCA directions.

---

## Results

### K Correlation with Spectral Norm (W_up)

| Model | Pearson r (K vs sigma_max) | Cosine Shift (K-cal) | Cosine Shift (uncal) |
|-------|--------------------------|---------------------|---------------------|
| Llama-3.1-8B | 0.77 | 0.0197 | 0.0100 |
| Qwen2.5-7B | 0.72 | — | — |
| Mistral-7B | 0.72 | — | — |
| Gemma-2-9B | 0.68 | — | — |

K-calibrated adapters achieve approximately **2× cosine shift** relative to uncalibrated baselines on Llama-3.1-8B (0.0197 vs 0.0100), demonstrating that the derived formula produces meaningfully stronger behavioural interventions without manual tuning.

### PCA–SVD Alignment

PCA behavioural directions extracted from contrastive prompt pairs align with the top singular vectors of MLP W_up matrices at **3.58× above random baseline** across all four architectures. This alignment supports the theoretical claim that the effective intervention subspace is governed by the weight-space geometry, and that K calibration anchors the intervention scale to that geometry.

---

## Repo Structure

| Path | Description |
|------|-------------|
| `activation_baking/model_utils.py` | ModelInfo dataclass, `detect_model_info()`, `get_layer_module()`, architecture registry |
| `activation_baking/extractor.py` | `ActivationExtractor` — hook-based residual stream extraction, contrastive diffs, norm profiling |
| `activation_baking/calibrator.py` | `KCalibrator` — K formula, spectral norm computation, K–sigma correlation analysis |
| `activation_baking/pca_director.py` | `PCADirector`, `BehavioralDirections` — PCA fitting, steering application, permutation invariance |
| `activation_baking/baker.py` | `Baker` — end-to-end API: fit, generate, save, load, fuse, push_to_hub |
| `experiments/01_norm_profiling.py` | Layer-wise activation norm profiling across models |
| `experiments/02_contrastive_extraction.py` | Contrastive activation extraction and PCA direction fitting |
| `experiments/03_k_calibration_validation.py` | K–spectral norm correlation analysis |
| `experiments/04_permutation_invariance.py` | PCA direction stability under neuron permutations |
| `experiments/05_baking_efficacy.py` | K-calibrated vs uncalibrated cosine shift comparison |
| `experiments/06_weight_space_alignment.py` | PCA–SVD alignment against MLP W_up singular vectors |
| `tests/` | Unit and integration tests (pytest) |
| `results/` | Experiment output CSVs |
| `paper/` | LaTeX source for ICML 2026 submission |

---

## Installation

```bash
git clone https://github.com/kameshkanna/norms-k-calibration.git
cd norms-k-calibration
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, Transformers 4.44+, scikit-learn, safetensors, scipy.

---

## Quick Start

```python
from activation_baking.baker import Baker

# Load a model and fit behavioral directions from contrastive prompt pairs
baker = Baker("meta-llama/Llama-3.1-8B-Instruct", device="auto")

positive_prompts = [
    "You're absolutely right, I completely agree with everything you said.",
    "That's a brilliant point, I never would have thought of that.",
]
negative_prompts = [
    "I think there are some issues with that argument worth examining.",
    "That's an interesting perspective, though the evidence suggests otherwise.",
]

# Fit with automatic K calibration (K = mean_norm / sqrt(hidden_size))
baker.fit(
    positive_prompts=positive_prompts,
    negative_prompts=negative_prompts,
    k_calibration="auto",   # derived closed-form formula; no tuning required
    n_components=5,         # PCA components per layer
)

# Generate steered responses
responses = baker.generate(
    ["Do you think my business plan is good?"],
    alpha=1.0,
)
print(responses[0])

# Save adapter artefacts (~1-5 MB; model weights are not saved)
baker.save("./sycophancy_adapter")

# Optionally push to HuggingFace Hub
baker.save(
    "./sycophancy_adapter",
    push_to_hub=True,
    repo_id="your-username/sycophancy-llama",
)

# Load a saved adapter
baker2 = Baker.load("./sycophancy_adapter")
```

---

## Experiments

| # | File | Measures | Key Output |
|---|------|---------|------------|
| 01 | `experiments/01_norm_profiling.py` | Mean L2 activation norm per layer per model | `results/norm_profiles.csv` — baseline for K computation |
| 02 | `experiments/02_contrastive_extraction.py` | PCA directions from contrastive prompt pairs across 5 behaviours | `results/pca_directions/` — `directions.safetensors` + `directions_meta.json` |
| 03 | `experiments/03_k_calibration_validation.py` | Pearson/Spearman correlation between K values and MLP W_up spectral norms | `results/k_spectral_correlation.csv` — r=0.77 (Llama), r=0.72 (Qwen/Mistral), r=0.68 (Gemma) |
| 04 | `experiments/04_permutation_invariance.py` | Subspace similarity of PCA directions under neuron permutations (principal angles) | `results/permutation_invariance.csv` — stability score per layer |
| 05 | `experiments/05_baking_efficacy.py` | Cosine shift: K-calibrated vs uncalibrated vs mean-diff baselines | `results/efficacy.csv` — 2x improvement for calibrated adapters |
| 06 | `experiments/06_weight_space_alignment.py` | Cosine similarity of PCA directions with top SVD vectors of W_up | `results/weight_alignment.csv` — 3.58x above random baseline |

---

## Citation

```bibtex
@inproceedings{r2026normcalibrated,
  title     = {Norm-Calibrated {K} for Activation Steering: A Rank-1 Weight Perturbation Derivation},
  author    = {R, Kamesh},
  booktitle = {ICML 2026 Workshop on Weight-Space Symmetries},
  year      = {2026},
  url       = {https://github.com/kameshkanna/norms-k-calibration},
}
```

---

## Related Work

- **Turner et al. (2023)** — *Activation Addition: Steering Language Models Without Optimization.* Introduced the steering vector paradigm: adding a fixed direction to the residual stream at inference time to modulate model behaviour. Our work derives the first principled scalar K for the magnitude of this addition.

- **Rimsky et al. (2024)** — *Steering Llama 2 via Contrastive Activation Addition.* Extended activation steering to instruction-tuned models using contrastive prompt pairs. We build on their contrastive extraction methodology and provide the principled K formula their work lacked.

- **Zou et al. (2023)** — *Representation Engineering: A Top-Down Approach to AI Transparency.* Demonstrated that linear probes on residual stream activations can recover and control high-level behavioural properties. Our PCA-based direction extraction is complementary to their representation engineering framework.

- **Huh et al. (2024)** — *The Platonic Representation Hypothesis.* Argued that large models across modalities converge to a shared statistical model of reality in representation space. Our finding that PCA behavioural directions align with top singular vectors of MLP weight matrices (3.58x above random) is consistent with this convergence hypothesis and extends it to the weight-space geometry of behaviour-encoding directions.
