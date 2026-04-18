# Norm-Calibrated K for Activation Steering

Principled derivation of steering strength **K = μ/√d** from a rank-1 weight-perturbation equivalence argument, validated across four transformer architectures.

## Key results

- K correlates with spectral norm of MLP weight matrices: Pearson r = 0.77 (Llama-3.1-8B), r = 0.72 (Qwen2.5-7B / Mistral-7B)
- PCA behavioural directions align with top singular vectors of MLP weight matrices at **3.58× above random baseline**
- K-calibrated adapters achieve **~2× cosine shift improvement** over uncalibrated steering (0.0197 vs 0.0100)
- Validated on Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B

## Structure

```
activation_baking/     core library (extractor, PCA director, calibrator, baker)
experiments/
  01_norm_profiling.py           layer-wise activation norm profiling
  02_contrastive_extraction.py   contrastive PCA direction extraction
  03_k_calibration_validation.py K formula validation + spectral norm correlation
  04_permutation_invariance.py   permutation invariance of directions
  05_baking_efficacy.py          K-calibrated vs uncalibrated efficacy comparison
  06_weight_space_alignment.py   PCA direction alignment with weight singular vectors
results/               experiment CSVs and plots
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from activation_baking.baker import Baker

baker = Baker(model_id="meta-llama/Llama-3.1-8B-Instruct", device="cuda")
baker.fit(positive_prompts, negative_prompts, k_calibration="auto")
outputs = baker.generate(prompts)
baker.save("my_adapter/")
```
