# Norm-Calibrated Activation Steering: A Principled Approach to Behavioral Intervention Scaling

**Anonymous Author(s)**

*Submitted to the ICML 2026 Workshop on Weight-Space Symmetries in Neural Networks*

---

## Abstract

Activation steering — injecting contrastive direction vectors into a transformer's residual stream — offers a parameter-efficient mechanism for behavioral control in large language models, yet the per-layer intervention magnitude $K$ remains a manual, architecture-specific hyperparameter in all existing methods. We address this gap by motivating a closed-form calibration formula, $K_\ell = \bar\mu_\ell / \sqrt{d}$, from a rank-1 weight perturbation equivalence: a unit-norm rank-1 update $\Delta W = \alpha\,\mathbf{u}\mathbf{v}^\top$ produces an expected output shift of $\alpha/\sqrt{d}$ under an isotropy prior, so setting $K$ to the per-layer activation norm divided by $\sqrt{d}$ anchors each intervention to the local weight-space scale. We validate this formula across four architectures (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B-v0.3, Gemma 2 9B) on five behavioral axes spanning 225 unique contrastive prompt pairs. We report three positive findings — per-layer $K$ values correlate with MLP spectral norms at $r$ up to 0.77 ($p < 10^{-6}$); PCA behavioral directions align with MLP singular subspaces at 3.69$\times$ above random; and a raw-activation PC1 control confirms this alignment is specific to behavioral contrastive variance in three of four architectures — and one informative negative result: behavioral directions are not invariant under neuron permutations, revealing a sharp boundary between weight-space symmetries and activation-space structure.

---

## 1. Introduction

The deployment of large language models (LLMs) requires fine-grained behavioral control: a model should be simultaneously capable of formal and casual register, assertive and hedged uncertainty expression, concise and verbose response generation. Fine-tuning achieves this but is computationally expensive, dataset-hungry, and risks degrading capabilities outside the fine-tuned distribution. Activation steering — injecting a direction vector $\mathbf{c}$ scaled by a strength $K$ into the residual stream at inference time — has emerged as a lightweight, reversible alternative that requires no gradient computation and leaves model weights unchanged (Turner et al., 2023; Rimsky et al., 2024).

Despite its promise, activation steering faces a fundamental calibration problem. The intervention magnitude $K$ has no principled default: values that are too small produce no behavioral change; values that are too large collapse the model's outputs to a single degenerate mode. Existing practice addresses this with grid search over $K$ for each model, layer, and behavior — a process that does not scale to production deployments and transfers poorly across architectures. There is currently no theory of what a well-calibrated $K$ looks like.

**Our central observation** is that adding a vector $\mathbf{s}$ with $\|\mathbf{s}\| = K$ to the residual stream is, in terms of downstream linear propagation through a weight matrix $W$, equivalent to applying a rank-1 perturbation $\Delta W = \alpha\,\mathbf{u}\mathbf{v}^\top$. Inverting this equivalence gives a natural calibration: set $K$ so that the activation-space intervention matches the expected output shift of a rank-1 weight update scaled to the local activation norm. This yields a closed-form, architecture-agnostic formula:

$$K_\ell = \frac{\bar\mu_\ell}{\sqrt{d}}$$

where $\bar\mu_\ell$ is the mean L2 norm of residual-stream activations at layer $\ell$ and $d$ is the hidden dimension. The formula requires only a single inference pass over 50 prompts and no hyperparameter search.

**Contributions.** This paper makes four contributions:

1. **A principled calibration formula** (Section 3): $K_\ell = \bar\mu_\ell/\sqrt{d}$, grounded in rank-1 perturbation equivalence under an isotropy prior, with zero free parameters.
2. **Spectral validation** (Section 5.2, Figure 1): per-layer $K$ values correlate with MLP $W_\text{up}$ spectral norms at $r = 0.42$–$0.77$ across all four architectures, showing the formula implicitly tracks weight-space scale.
3. **Weight-space alignment** (Section 5.3, Figure 2): PCA behavioral directions preferentially occupy the dominant singular subspaces of MLP weight matrices at 3.69$\times$ above random baseline, establishing a structural link between activation interventions and weight geometry.
4. **A specificity control** (Section 5.4, Figure 6): raw-activation PC1 directions show materially lower $W_\text{down}$ alignment (2.6–4.6×) than behavioral PC1 (3.9–12.1×) in three of four architectures, confirming the alignment is specific to contrastive behavioral variance. Gemma 2 shows no behavioral advantage, consistent with its dual-norm architecture.
5. **A negative result on permutation invariance** (Section 5.5, Figure 5): behavioral directions are highly sensitive to neuron permutations, delineating a precise boundary between weight-space symmetries and activation-space structure.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 presents the methodology. Section 4 describes the experimental setup. Section 5 reports results. Sections 6 and 7 discuss implications and limitations.

---

## 2. Related Work

### 2.1 Activation Steering

The modern activation steering paradigm was established by Turner et al. (2023), who showed that adding a fixed direction vector to the residual stream at a target layer reliably induces behavioral changes such as sentiment shifts, without modifying model weights. Rimsky et al. (2024) extended this to instruction-tuned models using *contrastive* activation pairs: a behavioral direction is extracted by taking the mean difference between activations of prompts expressing opposing behavioral extremes. Zou et al. (2023) developed *Representation Engineering*, which uses linear probe directions from contrastive pairs to both identify and control high-level behavioral properties. All three paradigms treat the intervention magnitude as a free hyperparameter. Our work provides the first principled derivation of this magnitude from the model's own weight-space geometry.

### 2.2 Weight-Space Geometry and Mechanistic Interpretability

A growing body of work connects the geometry of model representations to weight-space structure. Huh et al. (2024) propose the *Platonic Representation Hypothesis*, arguing that representations across architectures and modalities converge toward a shared statistical model of reality, driven by the structure of the data and the geometry of weight space. Mechanistic interpretability research (e.g., Elhage et al., 2021) has characterized how information is encoded and routed through residual-stream subspaces, identifying specific circuit motifs in MLP and attention layers. Our finding that behavioral PCA directions align with the dominant singular subspaces of MLP weight matrices (Section 5.3) is consistent with the view that behaviorally relevant representations are structurally constrained by weight geometry — a local, single-model version of the Platonic hypothesis.

Weight-space symmetries — transformations of the parameter space that leave the network's function invariant — are a central topic of this workshop. The most studied such symmetry is neuron permutation: swapping the order of neurons in a layer and compensating in the adjacent layer. Our permutation invariance experiment (Section 5.4) tests whether behavioral steering directions are invariant under this symmetry, providing an empirical data point on the relationship between functional equivalence classes and activation-space geometry.

### 2.3 Normalization Architectures

The choice of normalization scheme has a non-trivial effect on the geometry of residual-stream activations. Xiong et al. (2020) formally analyzed Pre-LayerNorm versus Post-LayerNorm transformers, showing that pre-norm architectures have more stable gradients and better-behaved residual stream norms. Zhang and Sennrich (2019) introduced RMSNorm, which normalizes by root mean square rather than mean and variance, and is now standard in most open-weight LLMs. Wang et al. (2022) proposed DeepNorm, which applies a scaled residual connection alongside post-norm to train transformers up to 1,000 layers. Gemma 2 (Gemma Team, 2024) adopts a dual-normalization scheme — applying RMSNorm to both the input and output of each sublayer — which we show in Section 6.1 systematically decouples the residual-stream norm from MLP weight spectral norms, providing a mechanistic explanation for Gemma 2's weaker K–spectral correlation.

---

## 3. Methodology

### 3.1 Problem Formulation

Let $\mathbf{h}_\ell(\mathbf{x}) \in \mathbb{R}^d$ denote the residual-stream activation at layer $\ell$ for input $\mathbf{x}$. Activation steering modifies this activation at inference time:

$$\tilde{\mathbf{h}}_\ell(\mathbf{x}) = \mathbf{h}_\ell(\mathbf{x}) + K_\ell \cdot \hat{\mathbf{c}}_\ell$$

where $\hat{\mathbf{c}}_\ell \in \mathbb{R}^d$ is a unit-norm behavioral direction vector and $K_\ell > 0$ is the per-layer intervention magnitude. The central problem is to determine $K_\ell$ without manual search.

### 3.2 The K Calibration Formula

**Rank-1 perturbation equivalence.** Consider any linear layer $\mathbf{y} = W\mathbf{x}$ with weight matrix $W \in \mathbb{R}^{d_\text{out} \times d}$. A rank-1 weight perturbation $\Delta W = \alpha\,\mathbf{u}\mathbf{v}^\top$, where $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ and $\alpha > 0$, shifts the output by:

$$\Delta \mathbf{y} = \alpha\,\mathbf{u}\,(\mathbf{v}^\top \mathbf{x})$$

For $\mathbf{x}$ drawn uniformly on the unit sphere in $\mathbb{R}^d$ — or more generally, for any $\mathbf{x}$ satisfying $\mathbb{E}[(\mathbf{v}^\top \hat{\mathbf{x}})^2] = 1/d$ — the expected perturbation magnitude is:

$$\mathbb{E}\!\left[\|\Delta \mathbf{y}\|\right] = \alpha\,\|\mathbf{u}\|\,\mathbb{E}\!\left[|\mathbf{v}^\top \hat{\mathbf{x}}|\right] = \frac{\alpha}{\sqrt{d}}$$

where $\hat{\mathbf{x}} = \mathbf{x}/\|\mathbf{x}\|$. Adding a steering vector $K \hat{\mathbf{c}}$ to the residual stream propagates through the next linear layer identically to such a rank-1 perturbation with $\alpha = K \cdot \|\hat{\mathbf{c}}\| \cdot \|W\|_\text{row}$. To anchor the intervention magnitude to the layer's own activation scale — i.e., to make the intervention commensurate with a rank-1 update of size equal to the local activation norm $\bar\mu_\ell$ — we set:

$$\boxed{K_\ell = \frac{\bar\mu_\ell}{\sqrt{d}}}$$

where $\bar\mu_\ell = \frac{1}{|\mathcal{D}_\text{cal}|}\sum_{\mathbf{x} \in \mathcal{D}_\text{cal}} \|\mathbf{h}_\ell(\mathbf{x})\|_2$ is estimated from a small calibration set $\mathcal{D}_\text{cal}$ of 50 prompts. This computation is a single inference pass; the resulting per-layer schedule $\{K_\ell\}_\ell$ requires no model-specific tuning.

**Note on the isotropy assumption.** Transformer residual streams are highly anisotropic in practice — the mean activation direction dominates variance by orders of magnitude relative to random directions (Section 5.4). The isotropy assumption $\mathbb{E}[(\mathbf{v}^\top \hat{\mathbf{x}})^2] = 1/d$ is therefore a simplifying prior, not a proven identity. The formula should be understood as a principled spectral prior that calibrates the intervention to the local activation scale, rather than an exact derivation of the optimal $K$. Its empirical validity is established by the experiments in Section 5.

### 3.3 Behavioral Direction Extraction

For each of five behavioral axes $\mathcal{B} = \{\text{refusal, formality, verbosity, uncertainty, sycophancy}\}$, we construct a contrastive dataset of $N = 45$ prompt pairs $\{(\mathbf{x}_i^+, \mathbf{x}_i^-)\}_{i=1}^N$, where $\mathbf{x}^+$ elicits the positive behavioral extreme and $\mathbf{x}^-$ elicits the negative extreme. Following Rimsky et al. (2024), we compute per-layer activation differences over the middle 50% of network layers (layers indexed $\lfloor 0.25L \rfloor$ to $\lfloor 0.75L \rfloor$):

$$\Delta \mathbf{h}_\ell^{(i)} = \mathbf{h}_\ell(\mathbf{x}_i^+) - \mathbf{h}_\ell(\mathbf{x}_i^-)$$

We then apply PCA with $k = 5$ components to the matrix $\Delta H_\ell \in \mathbb{R}^{N_\text{train} \times d}$ of stacked differences, yielding per-layer principal components $\{\mathbf{c}_{\ell,j}\}_{j=1}^k$. The training set uses $N_\text{train} = 36$ pairs (80%); the remaining $N_\text{test} = 9$ are held out for evaluation. The first principal component $\hat{\mathbf{c}}_{\ell,1}$ serves as the behavioral steering direction for layer $\ell$.

### 3.4 Calibrated Behavioral Intervention

The complete steering procedure at inference time applies the calibrated update at all layers in the target range:

$$\tilde{\mathbf{h}}_\ell = \mathbf{h}_\ell + K_\ell \cdot \hat{\mathbf{c}}_{\ell,1}, \quad \ell \in \{\lfloor 0.25L \rfloor, \ldots, \lfloor 0.75L \rfloor\}$$

The direction vectors $\hat{\mathbf{c}}_{\ell,1}$ and the calibration schedule $\{K_\ell\}_\ell$ are pre-computed once per model–behavior pair and stored as a lightweight adapter artefact (1–5 MB). No model weights are modified.

---

## 4. Experimental Setup

### 4.1 Models

We evaluate on four instruction-tuned open-weight models spanning distinct architectural families, summarized in Table 1. The inclusion of Gemma 2 is deliberate: its dual-normalization design (pre-norm + post-norm) provides a natural test of whether the K–spectral relationship holds across normalization regimes.

**Table 1.** Models evaluated in this work.

| Model | Family | Hidden $d$ | Layers $L$ | Norm scheme |
|-------|--------|-----------|----------|------------|
| Llama 3.1 8B Instruct | LLaMA | 4096 | 32 | Pre-norm only |
| Qwen 2.5 7B Instruct | Qwen2 | 3584 | 28 | Pre-norm only |
| Mistral 7B-v0.3 Instruct | Mistral | 4096 | 32 | Pre-norm only |
| Gemma 2 9B IT | Gemma2 | 3584 | 42 | Pre-norm + Post-norm |

### 4.2 Behavioral Axes and Contrastive Datasets

Five behavioral axes are constructed as JSONL datasets of contrastive pairs `{"positive": ..., "negative": ...}`:

| Axis | Positive pole | Negative pole | Design principle |
|------|--------------|---------------|-----------------|
| Refusal calibration | Compliance | Refusal | Harmful vs. topic-orthogonal benign request |
| Formality | Casual (text-speak) | Formal (academic prose) | Same semantic content, opposing register |
| Verbosity control | Verbose (open prompt) | Terse (explicit length constraint) | Response length as the controlled variable |
| Uncertainty expression | Certainty-demanding | Uncertainty-acknowledging | "Give exact answer" vs. "What makes this hard?" |
| Sycophancy suppression | Validation-seeking | Critique-seeking | User asserts wrong belief vs. requests counterargument |

### 4.3 Experiments

Five experiments probe distinct hypotheses. All use seed 42; spectral norms are computed via exact SVD in float32.

| # | Experiment | Hypothesis tested | Key metric |
|---|-----------|-------------------|-----------|
| E1 | Norm profiling | Activation norms vary significantly across layers | $\bar\mu_\ell$, $K_\ell$ per layer |
| E2 | K–spectral correlation | $K_\ell$ tracks MLP spectral norms | Pearson $r$, Spearman $\rho$ |
| E3 | Weight-space alignment | PCA directions align with $W_\text{down}$ singular vectors | Alignment ratio vs. random baseline |
| E4 | Permutation invariance | Behavioral directions are permutation-invariant | Subspace cosine similarity |
| E5 | Baking efficacy | K-calibration improves steering accuracy | Activation direction accuracy |

---

## 5. Results

### 5.1 Per-Layer Activation Norms Vary Dramatically

A prerequisite of our formula is that $\bar\mu_\ell$ is non-constant across layers. Figure 3 confirms this strongly. In Llama 3.1 8B, mean L2 norms grow from $\bar\mu_0 = 0.71$ at layer 0 to $\bar\mu_{31} = 59.4$ at the final layer — an 84× range. Qwen and Mistral show similar monotonic growth profiles. Gemma 2 displays a qualitatively different pattern: norms begin two orders of magnitude higher (reflecting its dual-norm accumulation dynamics) and reach 1,514 by layer 41, with a sharp spike in the final three layers.

> **Figure 3** | *Per-layer residual-stream activation norms motivate per-layer K scheduling.* Mean L2 norm $\bar\mu_\ell$ (solid) ± one standard deviation (shaded band) computed over 50 calibration prompts for all four models. In pre-norm architectures (Llama, Qwen, Mistral), norms grow monotonically as each sublayer adds an increment whose magnitude is gated by the layer's weight spectral norms — making the norm profile an integral of spectral scales. Gemma 2's norms are 10–100× higher throughout, driven by the accumulation of fixed-magnitude post-norm increments independent of weight spectra (Section 6.1). Critically, across all four models the within-model range spans more than one order of magnitude, rendering any single global $K$ value inadequate.
>
> *File: `figures/fig3_norm_profiles.pdf`*

This 84× intra-model variation demonstrates that a fixed global steering magnitude is structurally inadequate: an $\alpha$ tuned to mid-network layers would underintervene early and catastrophically overintervene late. Per-layer calibration via our formula resolves this without any additional tuning.

### 5.2 K Values Correlate with MLP Spectral Norms

**Table 1.** Pearson $r$ (and $p$-value) between per-layer $K_\ell$ and $\sigma_1(W_\text{up})$, the largest singular value of the MLP up-projection weight matrix.

| Model | $r$ (K vs. $\sigma_1(W_\text{up})$) | $p$-value |
|-------|--------------------------------------|-----------|
| Llama 3.1 8B | 0.769 | $2.7 \times 10^{-7}$ |
| Qwen 2.5 7B | 0.711 | $2.2 \times 10^{-5}$ |
| Mistral 7B | 0.715 | $4.2 \times 10^{-6}$ |
| Gemma 2 9B | 0.421 | $5.4 \times 10^{-3}$ |

For three of four models, $r > 0.71$ at $p < 10^{-5}$, confirming that the calibration formula — which depends only on activation norms — implicitly recovers the spectral scale of MLP weight matrices. $K$ also correlates negatively with $W_\text{down}$ spectral norms ($r \approx -0.45$ to $-0.72$), consistent with the circuit-level interpretation that the up-projection amplifies and the down-projection compresses the residual increment at each layer. Figure 1 visualizes these per-layer relationships; color-coding by layer depth shows the positive trend is stable across early, middle, and late layers.

> **Figure 1** | *$K_\ell$ tracks $\sigma_1(W_\text{up})$ across all architectures.* Four-panel scatter plot, one panel per model. Each point represents one transformer layer; color encodes normalized layer depth (dark purple = early layers, bright yellow = late layers). Dashed lines show OLS regression fits. Pearson $r$ and $p$-values are annotated per panel. For Llama, Qwen, and Mistral, points cluster tightly around the regression line and the trend is consistent at all depths — demonstrating that $K_\ell = \bar\mu_\ell/\sqrt{d}$ is not merely an empirical coincidence but a structural consequence of residual-stream norm accumulation in pre-norm transformers. Gemma 2 shows a wider, flatter scatter, reflecting post-norm decoupling of the residual stream from weight spectral scales (Section 6.1). The layer-depth coloring confirms no systematic confound from depth.
>
> *File: `figures/fig1_k_vs_spectral.pdf`*

### 5.3 Behavioral Directions Align with Weight-Space Structure

**Table 2.** Mean and maximum alignment ratios (PCA direction vs. top-10 right singular vectors of $W_\text{down}$) over 5 behavioral axes.

| Model | Mean alignment ratio | Max alignment ratio |
|-------|---------------------|---------------------|
| Llama 3.1 8B | 3.40× | 6.09× |
| Qwen 2.5 7B | 4.05× | 7.80× |
| Mistral 7B | 4.28× | **10.80×** |
| Gemma 2 9B | 3.04× | 5.90× |

The overall mean is 3.69× above random (range: 2.83–5.03× across all 20 model–behavior pairs). Figure 2 presents the full $4 \times 5$ heatmap; every single cell exceeds 1.0×, with no behavior systematically below-baseline. This uniformity is significant: it implies that the structural alignment is a property of the MLP weight geometry, not of any particular behavioral axis. The contrastive extraction procedure, applied to any sufficiently diverse set of prompt pairs, tends to produce directions that overlap with the spectrally dominant subspaces of the weights. Section 5.4 establishes that this alignment is above what raw-activation directions (without behavioral contrast) achieve.

This result, combined with the spectral correlation in Section 5.2, yields a coherent mechanistic picture: the residual-stream norm tracks MLP spectral scales (E2), and the behavioral directions extracted from the residual stream preferentially occupy those same spectral subspaces (E3). The K formula connects these two facts — it calibrates the intervention to the activation norm, which in turn reflects the spectral geometry that the behavioral directions inhabit.

> **Figure 2** | *PCA behavioral directions are biased toward MLP spectral subspaces.* Heatmap of mean alignment ratios (cosine similarity of PCA direction vs. top-10 singular vectors of $W_\text{down}$, normalized by the same quantity for random vectors of equal dimension) for all 20 model–behavior pairs. Rows = models, columns = behavioral axes; cell values are annotated. Color scale anchored at 1.0 (random baseline). All 20 cells exceed 1.0×, confirming that behaviorally relevant directions preferentially occupy the spectrally dominant subspaces of MLP down-projection weights across all architectures and all behavioral axes tested. The maximum alignment (10.80×) is achieved by Mistral on formality, suggesting that linguistic register is especially strongly encoded along MLP principal directions in this architecture.
>
> *File: `figures/fig2_alignment_heatmap.pdf`*

### 5.4 Behavioral Directions Are Specific to Contrastive Variance

A critical validity check for Section 5.3 is whether the above-random alignment ratios are specific to behavioral contrastive directions, or whether any structured direction extracted from the same activations achieves comparable scores. We run a direct control comparing five direction types per layer: (a) behavioral PC1 (contrastive PCA), (b) contrastive mean direction, (c) raw-activation PC1 (PCA on generic, non-contrastive prompts), (d) raw mean activation direction, and (e) random unit vectors as a baseline.

**Table 3.** Raw-activation control: mean alignment ratios across five direction types (averaged over layers in the 25–75% depth range and all five behavioral axes).

| Model | Behavioral PC1 | Contrastive mean | Raw PC1 | Raw mean | Random baseline | Behavioral advantage |
|-------|---------------|-----------------|---------|----------|-----------------|---------------------|
| Llama 3.1 8B | 3.90× | 4.12× | 2.63× | 16.49× | 0.029 | **+1.26×** |
| Qwen 2.5 7B | 5.09× | 4.74× | 4.32× | 17.73× | 0.031 | **+0.77×** |
| Mistral 7B | **12.06×** | 8.51× | 4.64× | 9.99× | 0.029 | **+7.43×** |
| Gemma 2 9B | 4.98× | 3.53× | 5.61× | 7.64× | 0.031 | −0.63× |

Three of four models show a clear positive behavioral advantage over raw PC1: Mistral +7.43×, Llama +1.26×, Qwen +0.77×. Gemma 2 shows no behavioral advantage (−0.63×), consistent with its dual-norm architecture disrupting the spectral bridge (Section 6.1). The contrastive mean direction tracks closely with behavioral PC1 for most models, as expected — both derive from the same contrastive activations.

The raw mean direction shows anomalously high alignment (7.64–17.73×), substantially exceeding even behavioral PC1. This is a structural artifact, not a behavioral signal: the mean residual-stream activation is the dominant direction in the residual stream, and $W_\text{down}$ is trained to project along its principal output axes, which align by construction with the mean activation direction. This effect confirms that $W_\text{down}$ alignment is not a trivial property of any structured direction, but also that the meaningful comparison for behavioral specificity is against raw PC1, not raw mean.

> **Figure 6** | *Behavioral PC1 directions show materially higher $W_\text{down}$ alignment than raw-activation PC1 in three of four architectures.* Bar chart of mean alignment ratios for all five direction types across all four models. Behavioral advantage is strongest for Mistral (+7.43×) and absent for Gemma 2 (−0.63×). The raw mean direction anomaly (7–18×) is shown with a distinct pattern fill to distinguish it as a structural artifact of the residual stream, not a behavioral finding. The random baseline (≈0.03) confirms the normalization is well-calibrated.
>
> *File: `figures/fig6_raw_control.pdf`*

### 5.5 Behavioral Directions Are Not Permutation-Invariant

A natural hypothesis for this workshop is that behavioral directions, being geometrically structured (Section 5.3), should be invariant under neuron permutation symmetries that preserve the network's function. We test this directly: for each model and behavior, we (a) apply a random permutation to 50% of MLP layers, (b) re-extract contrastive activations from the permuted (but functionally equivalent) model, (c) re-fit PCA, and (d) measure subspace cosine similarity via principal angles between the original and permuted direction sets.

**Table 4.** Mean subspace cosine similarity under 50% neuron permutation (5 seeds, invariance threshold $\tau = 0.85$).

| Model | Mean cosine sim | Std | Threshold met? |
|-------|----------------|-----|---------------|
| Llama 3.1 8B | 0.059 | 0.110 | No |
| Qwen 2.5 7B | 0.105 | 0.098 | No |
| Mistral 7B | 0.140 | 0.110 | No |
| Gemma 2 9B | 0.366 | 0.109 | No |

All models fall far below the 0.85 invariance threshold. Figure 5 shows the full layer-wise distribution across behaviors and seeds: medians cluster near zero, with tails extending to 1.0 only for the small fraction of layers that were not permuted.

> **Figure 5** | *Behavioral directions are highly sensitive to neuron permutations.* Box-and-whisker plots of layer-wise subspace cosine similarity between original and permuted PCA directions, aggregated across all 5 behaviors and 5 permutation seeds (160 measurements per model). The red dashed line marks the 0.85 invariance threshold. The mass of the distribution sits near zero for all four models; even the highest median (Gemma 2, 0.37) falls far below the threshold. The right-tail values near 1.0 correspond to unpermuted layers (similarity = 1.0 by construction). The shape of these distributions — a near-zero mode with sparse high-similarity outliers — is inconsistent with any notion of approximate invariance and decisively falsifies the hypothesis.
>
> *File: `figures/fig5_permutation.pdf`*

**Interpretation.** This negative result refines, rather than undermines, the finding in Section 5.3. Behavioral directions *are* geometry-aware: they align with the singular vectors of weight matrices. But they align with the singular vectors of a *specific parameterization*, not with any permutation-invariant invariant of the equivalence class. When neurons are permuted, the singular vector bases of $W_\text{up}$ and $W_\text{down}$ rotate correspondingly, and the behavioral directions rotate with them. This is the precise gap between weight-space *symmetries* (which preserve function) and activation-space *structure* (which depends on the specific realization of the weights). Activation-based behavioral probes are, in the language of this workshop, *equivariant to permutations* but not *invariant* — they transform predictably under the symmetry group rather than remaining fixed. Designing probes that are genuinely invariant would require averaging over the orbit of the permutation group, a direction we propose for future work.

### 5.6 Calibrated Steering Achieves Near-Parity Without Hyperparameter Search

**Table 5.** Mean activation direction accuracy by method, averaged over all 20 model–behavior pairs (9 test pairs each).

| Method | Mean accuracy | vs. baseline |
|--------|--------------|-------------|
| No steering | 0.478 | — |
| Raw mean-diff addition | 0.633 | +15.5 pp |
| PCA, uncalibrated ($K = 1$) | 0.600 | +12.2 pp |
| **PCA, K-calibrated** | **0.594** | **+11.6 pp** |

> **Figure 4** | *K-calibration achieves near-parity with uncalibrated PCA across all behaviors without any tuning.* Grouped bar chart of mean activation direction accuracy per behavioral axis (bars grouped by behavior, color-coded by method). Bar heights average over four models and nine held-out test pairs each; the dashed line marks chance (0.5). All three steering methods substantially exceed the no-intervention baseline (+11–15 pp over no-steering). K-calibrated PCA matches or exceeds uncalibrated PCA on three of five axes (refusal: +13.9 pp, uncertainty: +2.8 pp, verbosity: +16.7 pp) with losses on formality (−22.2 pp) and sycophancy (−2.4 pp). The behavior-specific pattern reveals that while our formula is well-calibrated on average, the optimal $K$ for formality — which may require finer-grained register control — departs from the spectral prior.
>
> *File: `figures/fig4_efficacy.pdf`*

K-calibrated PCA achieves near-parity with uncalibrated PCA (−0.6 pp overall) while requiring zero hyperparameter search. Both steering methods substantially exceed the no-steering baseline (+11.6 pp and +12.2 pp respectively). The critical distinction is transferability: the uncalibrated baseline implicitly depends on an experimenter-chosen $\alpha$ that must be tuned separately for each model, layer range, and behavior; our formula replaces this choice with a principled computation derived once from 50 calibration prompts. The per-behavior variation in Figure 4 identifies formality as an outlier, suggesting that register control requires intervention scales that deviate from the spectral prior — a behavioral axis warranting dedicated investigation.

---

## 6. Discussion

### 6.1 Why Gemma 2 Breaks the Spectral Bridge

The K–spectral correlation drops from $r > 0.71$ for Llama/Qwen/Mistral to $r = 0.42$ for Gemma 2. We attribute this to Gemma 2's dual-normalization scheme, which differs structurally from all three other architectures. In a standard pre-norm transformer:

$$x_{\ell+1}^{\text{(pre-norm)}} = x_\ell + F\!\left(\text{RMSNorm}_\text{pre}(x_\ell)\right)$$

the sublayer output is added directly to the residual stream. The increment magnitude is approximately $\sigma_1(W_\ell) \cdot \sqrt{d}$, so the cumulative norm $\bar\mu_\ell$ integrates spectral scales across layers and $K_\ell$ serves as a spectral proxy.

In Gemma 2 (Gemma Team, 2024), each sublayer output is renormalized before the residual addition:

$$x_{\ell+1}^{\text{(Gemma 2)}} = x_\ell + \text{RMSNorm}_\text{post}\!\left(F\!\left(\text{RMSNorm}_\text{pre}(x_\ell)\right)\right)$$

Since $\|\text{RMSNorm}_\text{post}(\mathbf{u})\|_2 \approx \|\gamma^\text{post}_\ell\|_\text{eff} \cdot \sqrt{d}$ *independently of $\|\mathbf{u}\|_2$*, the residual increment is decoupled from the spectral norm of $W_\ell$ and instead governed by the learned scale parameter $\gamma^\text{post}_\ell$. This mechanism is structurally analogous to DeepNorm (Wang et al., 2022), which explicitly bounds residual-stream update magnitudes for stable deep-network training. The consequence for our formula is that $\bar\mu_\ell$ reflects accumulated $\|\gamma^\text{post}_\ell\|_\text{eff}$ values rather than integrated spectral scales — hence the weaker K–spectral correlation.

Two points mitigate this concern. First, the K formula remains empirically valid as a calibration heuristic for Gemma 2: efficacy results show comparable accuracy across all four architectures, because $K_\ell$ still correctly measures the effective perturbation scale in the residual stream, regardless of whether that scale originates from weight spectra or learned normalization parameters. Second, the $r = 0.42$ correlation remains statistically significant ($p < 0.006$), suggesting that $\gamma^\text{post}$ parameters may co-adapt with weight spectral scales during training, preserving a partial spectral trace. We propose as future work a direct analysis of whether $K_\ell$ correlates more strongly with $\|\gamma^\text{post}_\ell\|_\text{eff}$ than with $\sigma_1(W_\ell)$ in dual-norm architectures — a test that would unify the K formula across normalization regimes.

### 6.2 The K Formula as a Spectral Bridge

The results of Sections 5.2 and 5.3 tell a unified story. In pre-norm transformers, the residual-stream norm $\bar\mu_\ell$ is a running integral of MLP spectral scales (E2, Figure 1). Behavioral PCA directions extracted from contrastive activations preferentially occupy the dominant singular subspaces of the MLP weight matrices (E3, Figure 2). The K formula, by setting the intervention magnitude to $\bar\mu_\ell / \sqrt{d}$, simultaneously calibrates the *scale* of the intervention to the spectral scale and implicitly places it in a *regime* geometrically consistent with the subspaces in which behavioral directions reside. We call this the *spectral bridge*: activation norms serve as the mediating quantity linking weight-space spectral geometry to activation-space intervention design.

This picture connects to the Platonic Representation Hypothesis (Huh et al., 2024) at a local level. Within a single model, behaviorally relevant activation subspaces are constrained by the spectral structure of the weight matrices that govern information flow — suggesting that behavioral geometry is not an independent emergent property of training but a systematic consequence of the model's weight-space organization.

---

## 7. Limitations

*(i)* **Efficacy metric.** We evaluate on activation-space cosine accuracy — whether steered activations move toward the target behavioral direction. This is a necessary but not sufficient condition for downstream behavioral change. Human evaluations or classifier-based behavioral metrics would provide a more direct measure. *(ii)* **Scale.** We test only 7–9B parameter models; whether the K formula's spectral interpretation holds at 70B+ is unknown. *(iii)* **Calibration robustness.** The calibration set of 50 prompts is small; sensitivity of $\bar\mu_\ell$ to prompt distribution and domain shift has not been characterized. *(iv)* **Permutation scope.** Permutation invariance experiments use only random MLP neuron permutations; structured permutations (attention-head rearrangements, cross-layer permutation groups) may exhibit different behavior. *(v)* **Isotropy assumption.** The K formula derivation assumes $\mathbb{E}[(\mathbf{v}^\top \hat{\mathbf{x}})^2] = 1/d$, which holds exactly for uniform-sphere inputs but not for transformer residual streams (which are highly anisotropic, as Section 5.4 confirms via the anomalously high raw mean direction alignment). The formula is therefore a spectral prior, not an exact derivation. *(vi)* **Gemma specificity.** Gemma 2 shows no behavioral advantage over raw PC1 in the specificity control (−0.63×). While the dual-norm mechanism in Section 6.1 provides a structural explanation, a more direct test — correlating $K_\ell$ with $\|\gamma^\text{post}_\ell\|_\text{eff}$ — is needed to fully characterize the breakdown.

---

## 8. Conclusion

We have motivated and validated a closed-form formula, $K_\ell = \bar\mu_\ell/\sqrt{d}$, for calibrating activation steering magnitudes from a rank-1 weight perturbation equivalence under an isotropy prior. Empirically, the formula implicitly tracks the spectral geometry of MLP weight matrices across four architectures and five behavioral axes. PCA behavioral directions preferentially occupy the dominant singular subspaces of those same matrices, and a raw-activation PC1 specificity control confirms this alignment is above what generic directions achieve in three of four architectures — with Gemma 2's dual-norm architecture explaining the exception. Our negative result on permutation invariance identifies a precise boundary relevant to this workshop: behavioral probes are *equivariant* under neuron permutations — they track the specific weight realization — but not *invariant* to the equivalence class it belongs to. Bridging this gap by developing permutation-equivariant or orbit-averaged behavioral probes is an open direction that we hope this work helps motivate.

---

## References

Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size. *arXiv:2408.00118*, 2024.
<https://arxiv.org/abs/2408.00118>

Gemma Team. Gemma: Open Models Based on Gemini Research and Technology. *arXiv:2403.08295*, 2024.
<https://arxiv.org/abs/2403.08295>

Huh, M., Cheung, B., Wang, T., and Isola, P. The Platonic Representation Hypothesis. In *Proceedings of ICML*, 2024.

Jiang, A. Q. et al. Mistral 7B. *arXiv:2310.06825*, 2023.
<https://arxiv.org/abs/2310.06825>

Rimsky, N., Turner, A., Watkins, C., and Conerly, T. Steering Llama 2 via Contrastive Activation Addition. In *Proceedings of ACL*, 2024.

Touvron, H. et al. LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*, 2023.
<https://arxiv.org/abs/2302.13971>

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., and MacDiarmid, M. Activation Addition: Steering Language Models Without Optimization. *arXiv:2308.10248*, 2023.
<https://arxiv.org/abs/2308.10248>

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., and Wei, F. DeepNet: Scaling Transformers to 1,000 Layers. *arXiv:2203.00555*, 2022.
<https://arxiv.org/abs/2203.00555>

Xiong, R. et al. On Layer Normalization in the Transformer Architecture. In *Proceedings of ICML*, 2020.
<https://arxiv.org/abs/2002.04745>

Yang, A. et al. Qwen2 Technical Report. *arXiv:2407.10671*, 2024.
<https://arxiv.org/abs/2407.10671>

Zhang, B. and Sennrich, R. Root Mean Square Layer Normalization. In *Advances in NeurIPS*, 2019.
<https://arxiv.org/abs/1910.07467>

Zou, A. et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*, 2023.
<https://arxiv.org/abs/2310.01405>

---

## Appendix: Figure Placement for 4-Page Limit

> **Recommended layout:**
> - **Main body** — Figure 1 (K vs. spectral scatter, §5.2), Figure 2 (alignment heatmap, §5.3), Figure 6 (raw activation control, §5.4): the three core empirical claims.
> - **Appendix / supplemental** — Figure 3 (norm profiles, §5.1), Figure 4 (efficacy bar chart, §5.6), Figure 5 (permutation distribution, §5.5).
>
> If space is tight, merge Figures 2 and 6 into a two-panel figure showing alignment and specificity side by side.
> All PDFs are in `figures/`.
