# Norm-Calibrated K for Activation Steering: A Rank-1 Weight Perturbation Derivation

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Activation steering applies constant direction vectors to transformer residual streams at inference time to shape model behaviour. The steering strength K is typically grid-searched per model and per layer, limiting transferability across architectures. This paper derives a closed-form formula for K from a rank-1 weight-perturbation equivalence and validates it empirically across four architectures and five behavioral axes.

---

## Method

### K-Calibration Formula

From the rank-1 perturbation equivalence (Proposition 1):

```
K_ℓ = μ̄_ℓ / √d
```

where `μ̄_ℓ` is the mean L2 norm of residual stream activations at layer `ℓ`, estimated from 50 calibration prompts in a single forward pass. No hyperparameter search required.

**Spectral Integral (Proposition 2).** In pre-norm transformers, `μ̄_ℓ ≈ μ̄_0 + Σ c_ℓ′ · σ₁(W_dn,ℓ′)`, making `K_ℓ` a running integral of MLP spectral norms and predicting strong correlation with `σ₁(W_up)`.

**Gemma-2 Post-Norm (Remark 1).** Gemma-2's dual RMSNorm clips each sublayer increment to `‖γᵖᵒˢᵗ‖ · √d`, decoupling `K_ℓ` from weight spectra. The formula `K_ℓ = μ̄_ℓ/√d` is unchanged; `μ̄_ℓ` integrates accumulated normalization scales instead.

---

## Results

Four instruction-tuned models (Llama 3.1-8B, Qwen 2.5-7B, Mistral-7B, Gemma 2-9B), five behavioral axes (refusal calibration, formality, verbosity control, uncertainty expression, sycophancy suppression), 45 contrastive pairs per axis, 80/20 train/test split, seed 42.

### E1 — Norm Profiles

| Model | K̄ (mean over layers) | μ̄ (mean activation norm) |
|---|---|---|
| Llama 3.1-8B | 0.249 | 15.9 |
| Mistral-7B | 0.151 | 9.7 |
| Qwen 2.5-7B | 1.889 | 120.8 |
| Gemma 2-9B | 6.488 | 388.2 |

Llama norms grow 84× from layer 0 to 31; Gemma reaches 1514 at layer 41. A fixed global K is structurally wrong across and within models.

### E2 — K–Spectral Correlation

| Model | r | p | Proxy |
|---|---|---|---|
| Llama 3.1-8B | 0.769 | 2.7×10⁻⁷ | σ₁(W_up) |
| Qwen 2.5-7B | 0.711 | 2.2×10⁻⁵ | σ₁(W_up) |
| Mistral-7B | 0.715 | 4.2×10⁻⁶ | σ₁(W_up) |
| Gemma 2-9B | r=0.960 (ρ=0.9992) | <10⁻²³ | Σ‖γᵖᵒˢᵗ‖ |

Partial correlation controlling for layer depth strengthens all results (Llama: 0.769→0.901; Qwen: 0.711→0.832; Mistral: 0.715→0.780; Gemma: 0.421→0.696), confirming the spectral grounding is not a depth confound.

### E3 — Weight-Space Alignment

PCA behavioral directions align with the top-10 right singular vectors of W_dn at **3.69× random baseline** (range 3.04–4.28×). All 20 model–behavior cells exceed 1.0×.

| Model | Mean ratio | Max ratio |
|---|---|---|
| Llama 3.1-8B | 3.40× | 6.09× |
| Qwen 2.5-7B | 4.05× | 7.80× |
| Mistral-7B | 4.28× | 10.80× |
| Gemma 2-9B | 3.04× | 5.90× |

### E5 — Permutation Sensitivity

Directions show significant variance under 50% neuron permutation: cosine similarity drops to 0.051–0.412 across models, all below the τ=0.85 invariance threshold. Directions are parameterization-specific rather than functional-class invariants.

### Directional Fidelity

Mean cosine shift (↑ better), averaged over 5 behaviors:

| Model | Baseline | Raw (K=1) | PC1 (K=1) | PC1 (K-cal.) |
|---|---|---|---|---|
| Llama 3.1-8B | 0.0040 | 0.0121 | 0.0014 | **0.0471** |
| Gemma 2-9B | 0.0033 | 0.0158 | 0.0117 | 0.0065† |
| Qwen 2.5-7B | 0.0003 | 0.0087 | 0.0119 | 0.0018 |
| Mistral-7B | -0.0069 | 0.0443 | 0.0281 | 0.0170 |

Llama: K-calibration gives +0.046 absolute (+3349%) — strongest signal because pre-norm residual norms span >10× within-model range. Gemma (†): slight loss predicted by Remark 1 (post-norm overshoots at K_ℓ ≈ 25).

---

## Repository Structure

```
norms_k_calibration_paper/
├── activation_baking/
│   ├── baker.py                          # End-to-end API: fit, generate, save, load
│   ├── pca_director.py                   # PCADirector: contrastive directions, permutation invariance
│   ├── calibrator.py                     # KCalibrator: K = mean_norm / sqrt(hidden_size)
│   ├── extractor.py                      # ActivationExtractor: hook-based residual stream extraction
│   └── model_utils.py                    # ModelInfo, detect_model_info, architecture registry
├── experiments/
│   ├── 01_norm_profiling.py              # Per-layer μ̄_ℓ and K_ℓ (E1)
│   ├── 02_contrastive_extraction.py      # PCA directions from contrastive pairs
│   ├── 03_k_calibration_validation.py    # K–spectral norm correlation (E2)
│   ├── E4_weight_alignment_v2.py         # PCA–SVD alignment with bootstrap CIs (E3)
│   ├── E5_permutation_invariance_v2.py   # Permutation sensitivity with orbit probe (E5)
│   ├── 04_permutation_invariance.py      # Permutation invariance (original)
│   ├── 05_baking_efficacy.py             # K-calibrated vs uncalibrated cosine shift
│   ├── 06_weight_space_alignment.py      # PCA–SVD alignment (original)
│   ├── 07_cross_arch_comparison.py       # Cross-architecture CKA
│   ├── E7_k_sensitivity_curve.py         # K sensitivity sweep
│   ├── 08_directional_fidelity_analysis.py   # Directional fidelity aggregate
│   └── 09_partial_correlation_analysis.py    # Depth-controlled partial correlation (E6)
├── tests/
│   ├── unit/
│   └── integration/
├── data/behaviors/        # 5 × 45-pair contrastive JSONL datasets
├── results/               # Experiment output CSVs (populated after run)
├── paper.tex              # Paper LaTeX source
├── mentor_report.tex      # Research progress report
└── run_all.sh             # Master experiment runner
```

---

## Installation

```bash
git clone https://github.com/kameshkanna/norms-k-calibration.git
cd norms-k-calibration
source setup.sh
```

**Requirements:** Python 3.9+, PyTorch 2.4+, Transformers 4.47+, scikit-learn, safetensors, scipy.

---

## Running Experiments

```bash
# Full suite (all 4 paper models)
bash run_all.sh --paper-models

# Single model
bash run_all.sh --model llama_8b

# From a specific stage
bash run_all.sh --from-stage 4

# Individual experiments
python experiments/01_norm_profiling.py --model all --device cuda
python experiments/03_k_calibration_validation.py --model all --device cuda
python experiments/E4_weight_alignment_v2.py --model all --device cuda
python experiments/09_partial_correlation_analysis.py
```

---

## Quick Start

```python
from activation_baking.baker import Baker

baker = Baker("meta-llama/Llama-3.1-8B-Instruct", device="auto")

positive_prompts = [
    "You're absolutely right, I completely agree.",
    "That's brilliant, I never would have thought of that.",
]
negative_prompts = [
    "I think there are issues with that argument worth examining.",
    "That's an interesting perspective, though the evidence suggests otherwise.",
]

baker.fit(positive_prompts, negative_prompts, k_calibration="auto")
responses = baker.generate(["Do you think my business plan is good?"], alpha=1.0)
print(responses[0])

baker.save("./sycophancy_adapter")
```

---

## Citation

```bibtex
@article{r2026normcalibrated,
  title   = {Norm-Calibrated {K} for Activation Steering: A Rank-1 Weight Perturbation Derivation},
  author  = {R, Kamesh},
  year    = {2026},
  url     = {https://github.com/kameshkanna/norms-k-calibration},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
