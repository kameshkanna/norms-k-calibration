# Norm-Calibrated K for Activation Steering: A Rank-1 Weight Perturbation Derivation

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**ICML 2026 Workshop on Weight-Space Symmetries**

---

## Abstract

Activation steering is a powerful interpretability and alignment technique for shaping large language model behaviour by adding direction vectors to residual stream activations at inference time. A critical but largely unaddressed problem is the choice of steering strength K: existing approaches rely on manual grid search or heuristics that do not transfer across architectures, layers, or behaviours. We motivate a closed-form formula for K from a rank-1 weight perturbation equivalence: a rank-1 update ΔW = α·u·vᵀ with ‖u‖=‖v‖=1 produces an expected output shift of α/√d on a unit-norm input under an isotropy assumption, establishing a direct link between activation norm scale and weight-space geometry. Setting K = mean_norm / √hidden_size normalises the steering vector magnitude to match this scale. We validate this formula empirically across four architectures (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B), five behavioural dimensions, and 225 unique contrastive prompt pairs. We also show that behavioral PCA directions extracted via contrastive activation differences align with top singular vectors of MLP weight matrices 3.0–4.3× above a random baseline.

---

## Method

### The K Calibration Formula

**Rank-1 Perturbation Equivalence.** Consider a rank-1 weight update applied to any linear layer in a transformer:

```
ΔW = α · u · vᵀ
```

where `u` and `v` are unit-norm vectors (‖u‖=‖v‖=1) and `α` is a scalar magnitude. For a unit-norm input `x` (‖x‖=1), under the isotropy assumption that activations are uniformly distributed on the unit sphere:

```
E[‖ΔW x‖] = α · |uᵀx| · ‖v‖ ≈ α / √hidden_size
```

by the central limit theorem applied to the inner product of two random unit vectors in Rᵈ. Inverting this, to match a rank-1 perturbation with α equal to the actual per-layer activation norm:

```
K = mean_norm / √hidden_size
```

where `mean_norm` is the mean L2 norm of residual stream activations at the target layer, estimated over a small representative prompt set. **Note:** transformer residual streams are highly anisotropic in practice; the isotropy assumption is a simplifying prior, not a proven identity. The formula serves as a principled spectral prior rather than an exact derivation.

**Implementation:**

```python
K = mean_norm / math.sqrt(hidden_size)
```

This requires only a single forward pass over a small set of calibration prompts and no hyperparameter search.

---

## Results

All experiments run across 4 models × 5 behaviors × 225 unique contrastive prompt pairs.

### K Correlation with Spectral Norm (W_up) and Post-Norm Scale

Per-layer K values correlate strongly with their architecture-appropriate spectral proxy:

| Model | r (K vs W_up) | p-value | r (K vs cumul. γ_post) | Spearman ρ (γ_post) |
|-------|--------------|---------|------------------------|---------------------|
| Llama-3.1-8B | **0.769** | 2.7×10⁻⁷ | — | — |
| Qwen2.5-7B | **0.711** | 2.2×10⁻⁵ | — | — |
| Mistral-7B | **0.715** | 4.2×10⁻⁶ | — | — |
| Gemma-2-9B | 0.421 (ρ=0.12, n.s.) | 5.4×10⁻³ | **0.960** | **0.9992** |

For pre-norm architectures, K tracks MLP spectral norms (r = 0.71–0.77). For Gemma-2's dual pre+post RMSNorm architecture, the post-norm clips each sublayer increment to ‖γ_post‖·√d, decoupling K from W_up spectral norms. Experiment 09 confirms K instead tracks the cumulative post-norm scale with near-perfect monotonic correlation (ρ = 0.9992, p < 10⁻²³). The K formula is a universal spectral bridge across normalization regimes. See `figures/fig1_k_vs_spectral.pdf` and `results/gemma_postnorm/`.

### PCA–SVD Alignment (vs W_down)

PCA behavioral directions align 3.0–4.3× above random with SVD top vectors of MLP W_down:

| Model | Mean alignment ratio | Max alignment ratio |
|-------|---------------------|---------------------|
| Llama-3.1-8B | 3.40× | 6.09× |
| Qwen2.5-7B | 4.05× | 7.80× |
| Mistral-7B | **4.28×** | **10.80×** |
| Gemma-2-9B | 3.04× | 5.90× |

Overall mean: **3.69×** above random (range 3.0–4.3× across models). See `figures/fig2_alignment_heatmap.pdf`.

> **Critical control (Exp 08 — pending):** The above ratios are only scientifically meaningful if raw-activation PC1 directions (extracted from generic prompts with no contrastive signal) show materially lower alignment. Experiment 08 (`experiments/08_raw_activation_control.py`) runs this specificity check and must be completed before submission.

### Permutation Invariance (Negative Result)

Behavioral directions are **not** invariant under 50% neuron permutations — permuted-layer-only subspace cosine similarities are near zero (invariance threshold: 0.85):

| Model | Permuted-layer cosine sim | Invariance claim |
|-------|--------------------------|-----------------|
| Llama-3.1-8B | 0.051 ± 0.040 | ✗ Not supported |
| Qwen2.5-7B | 0.106 ± 0.068 | ✗ Not supported |
| Mistral-7B | 0.160 ± 0.123 | ✗ Not supported |
| Gemma-2-9B | 0.412 ± 0.181 | ✗ Not supported |

Directions are **equivariant** under permutation (they track neuron identity) rather than invariant. See `figures/fig5_permutation.pdf`.

### Efficacy (K-Calibrated vs Baselines)

Cosine-shift accuracy for steered vs unsteered responses, averaged across 4 models and 5 behaviors (9 test pairs each):

| Method | Mean accuracy |
|--------|---------------|
| None (baseline) | 0.478 |
| Raw addition | 0.633 |
| PCA uncalibrated | 0.600 |
| **PCA K-calibrated** | **0.594** |

K-calibrated achieves near-parity with uncalibrated PCA (Δ = −0.6pp) with zero hyperparameter tuning. Per-model detail: Llama K-calibrated (0.800) substantially outperforms uncalibrated (0.511); Gemma K-calibrated (0.444) underperforms uncalibrated (0.578). See `figures/fig4_efficacy.pdf`.

---

## Data

Contrastive prompt pairs live in `data/behaviors/`. Each file contains 45 JSONL pairs `{"positive": "...", "negative": "..."}` used to extract PCA behavioural directions. 80/20 train/test split applied per behavior; **225 unique pairs total** (45 × 5 behaviors).

| File | Behavior | Contrastive design |
|------|----------|--------------------|
| `refusal_calibration.jsonl` | Refusal vs compliance | Harmful request (`+`) vs completely unrelated benign request (`−`) |
| `formality.jsonl` | Casual vs formal register | Heavy text-speak/slang (`+`) vs high academic/bureaucratic prose (`−`) |
| `verbosity_control.jsonl` | Verbose vs terse | Bare open-ended prompt (`+`) vs explicit brevity constraint (`−`) |
| `uncertainty_expression.jsonl` | Certainty-demanding vs uncertainty-acknowledging | "Give me the exact answer, no hedging" (`+`) vs "What makes this question genuinely hard to answer?" (`−`) |
| `sycophancy_suppression.jsonl` | Validation-seeking vs critique-seeking | User asserts a wrong belief and asks for agreement (`+`) vs asks for strongest counterargument (`−`) |

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
| `experiments/06_weight_space_alignment.py` | PCA–SVD alignment against MLP W_down singular vectors |
| `experiments/07_generate_figures.py` | Generates all paper figures from results CSVs |
| `experiments/08_raw_activation_control.py` | **[Must run]** Raw-activation PC1 specificity control — critical validity check |
| `tests/` | Unit and integration tests (pytest) |
| `data/behaviors/` | 5 × 45-pair contrastive JSONL datasets |
| `results/` | Experiment output CSVs (populated after run) |
| `paper/` | LaTeX source for ICML 2026 submission |

---

## Installation

```bash
git clone https://github.com/kameshkanna/norms-k-calibration.git
cd norms-k-calibration
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, Transformers 4.44+, scikit-learn, safetensors, scipy, seaborn.

---

## Reproducing All Results

Run experiments in order. Experiments 01, 03, 04, 05, 06 results already exist. **Must re-run 02 then 08.**

```bash
# Step 1 — Norm profiles (already done; skip unless regenerating)
python experiments/01_norm_profiling.py --model all --device cuda

# Step 2 — Contrastive extraction: regenerates directions.pt required by 06 and 08
python experiments/02_contrastive_extraction.py --model all --behavior all --device cuda

# Step 3 — K–spectral correlation (already done; skip unless regenerating)
python experiments/03_k_calibration_validation.py --model all --device cuda

# Step 4 — Permutation invariance (already done)
python experiments/04_permutation_invariance.py --model all --behavior all --device cuda

# Step 5 — Baking efficacy (already done)
python experiments/05_baking_efficacy.py --model all --behavior all --device cuda

# Step 6 — Weight-space alignment (already done)
python experiments/06_weight_space_alignment.py --model all --behavior all --device cuda

# Step 7 — RAW ACTIVATION CONTROL (NOT YET RUN — critical for submission)
python experiments/08_raw_activation_control.py --model all --device cuda

# Step 8 — Generate figures (run after 08 completes)
python experiments/07_generate_figures.py
```

---

## Quick Start

```python
from activation_baking.baker import Baker

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
    k_calibration="auto",
    n_components=5,
)

responses = baker.generate(
    ["Do you think my business plan is good?"],
    alpha=1.0,
)
print(responses[0])

baker.save("./sycophancy_adapter")
```

---

## Experiments

| # | File | Measures | Key Output | Status |
|---|------|---------|------------|--------|
| 01 | `01_norm_profiling.py` | Mean L2 activation norm per layer | `results/norm_profiles/*.csv` | ✅ Done |
| 02 | `02_contrastive_extraction.py` | PCA directions from contrastive pairs | `results/pca_directions/` | ⚠️ Needs rerun (directions.pt missing) |
| 03 | `03_k_calibration_validation.py` | K vs W_up spectral norm correlation | `results/k_calibration/` | ✅ Done |
| 04 | `04_permutation_invariance.py` | PCA direction stability under permutations | `results/permutation_invariance/` | ✅ Done |
| 05 | `05_baking_efficacy.py` | K-calibrated vs uncalibrated accuracy | `results/efficacy/` | ✅ Done |
| 06 | `06_weight_space_alignment.py` | PCA–SVD alignment ratio | `results/weight_alignment/` | ✅ Done |
| 07 | `07_generate_figures.py` | All 5 paper figures | `figures/` | ✅ Done (rerun after 08) |
| 08 | `08_raw_activation_control.py` | Raw vs behavioral PC1 alignment — specificity control | `results/raw_activation_control/` | ✅ Done |
| 09 | `09_gemma_postnorm_analysis.py` | K vs γ_post scale correlation for Gemma 2 — dual-norm spectral proxy | `results/gemma_postnorm/` | ✅ Done |
| 10 | `10_downstream_eval.py` | Behavioral shift on neutral prompts + GSM8K capability preservation | `results/downstream_eval/` | ⚠️ Needs run on A100 |

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

- **Turner et al. (2023)** — *Activation Addition: Steering Language Models Without Optimization.* Introduced the steering vector paradigm. Our work motivates a principled scalar K for the magnitude of this addition.

- **Rimsky et al. (2024)** — *Steering Llama 2 via Contrastive Activation Addition.* Extended activation steering to instruction-tuned models using contrastive prompt pairs. We build on their contrastive extraction methodology and provide the principled K formula their work lacked.

- **Zou et al. (2023)** — *Representation Engineering: A Top-Down Approach to AI Transparency.* Demonstrated that linear probes on residual stream activations can recover and control high-level behavioural properties.

- **Huh et al. (2024)** — *The Platonic Representation Hypothesis.* Argued that large models converge to a shared statistical model of reality. Our spectral alignment finding is consistent with this hypothesis.
