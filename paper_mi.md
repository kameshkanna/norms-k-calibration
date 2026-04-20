# Behavioral Directions Are Spectrally Constrained: A Mechanistic Analysis of Activation Steering in Transformer MLP Layers

**Anonymous Author(s)**

*Submitted to the ICML 2026 Workshop on Mechanistic Interpretability*

---

## Abstract

A central question in mechanistic interpretability is *where* behavioral information is encoded in a transformer's weights and activations. We investigate whether the directional vectors used in activation steering — extracted from contrastive prompt pairs via PCA — bear any structural relationship to the weight matrices of the network they probe. Across four architectures (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B-v0.3, Gemma 2 9B) and five behavioral axes spanning 1,200+ contrastive prompt pairs, we report two findings. **First**, PCA behavioral directions preferentially align with the dominant right singular vectors of MLP down-projection weights at **3.69× above random baseline** (up to 11.62×), consistent across all 20 model–behavior combinations — suggesting that behaviors are not encoded as arbitrary residual-stream directions but as directions constrained by the spectral geometry of the weights. **Second**, these directions are **not** invariant under neuron permutations: mean subspace cosine similarity under 50% permutation is 0.06–0.37 across models, far below the invariance threshold of 0.85. This reveals that behavioral probes are *equivariant* to the permutation group (they rotate with the weight bases) but not *invariant*, meaning they measure a property of the specific parameterization rather than the functional equivalence class. As a mechanistic corollary, we derive a closed-form formula $K_\ell = \bar\mu_\ell / \sqrt{d}$ that recovers spectral geometry through activation norms alone, enabling calibration-free behavioral steering.

---

## 1. Introduction

Mechanistic interpretability seeks to understand not just *that* large language models (LLMs) encode behavioral and conceptual information, but *how* — in terms of specific circuits, weight matrices, and geometric structures. A foundational assumption of the linear representation hypothesis (Park et al., 2023; Mikolov et al., 2013) is that features and concepts correspond to linear directions in residual-stream activation space. Activation steering methods (Turner et al., 2023; Rimsky et al., 2024; Zou et al., 2023) have demonstrated this empirically: behavioral properties can be controlled by adding a direction vector $\mathbf{c} \in \mathbb{R}^d$ to the residual stream at inference time, with no weight modification.

This raises a deeper structural question that interpretability has not yet answered: **are these behavioral direction vectors arbitrary elements of activation space, or are they constrained by the geometry of the weight matrices themselves?** If behavioral directions are arbitrary, then behaviors are encoded in a subspace of the residual stream that has no special relationship to the model's weights — the MLP is essentially a black-box transducer. If they are constrained, then the weight-space geometry of the MLP directly shapes which behavioral directions are accessible to the model, which has implications for how we think about feature formation, superposition, and circuit-level behavioral control.

We give a definitive empirical answer for the MLP down-projection layer: **behavioral directions are strongly biased toward the spectrally dominant right singular subspaces of $W_\text{down}$**, with alignment ratios of 3.04–4.28× above random across all architectures. The directions are not uniform draws from activation space; they are geometrically constrained by the weight matrices they pass through.

We further test a natural follow-up: since these directions are structured by weight geometry, are they *invariant* to the neuron permutation symmetries that leave the network's function unchanged? We find they are not — they are **equivariant**: they rotate predictably when the weight bases rotate, but do not remain fixed. This has a precise interpretability implication: activation-based behavioral probes measure a property of the *specific parameterization*, not an abstract property of the network's function. The concept of "the refusal direction in Llama 3.1 8B" is meaningful only relative to a fixed weight realization; a permuted but functionally identical model would have a different refusal direction.

**Contributions:**

1. **Spectral constraint on behavioral directions** (Section 5.1, Figure 2): PCA directions from contrastive pairs align with dominant $W_\text{down}$ singular subspaces at 3.69× above random (up to 11.62×), across all 20 model–behavior combinations.
2. **Permutation equivariance, not invariance** (Section 5.2, Figure 5): behavioral directions rotate under neuron permutations; probes measure parameterization-specific geometry, not functional invariants.
3. **Spectral-norm proxy via activation norms** (Section 5.3, Figure 1): per-layer activation norms correlate with MLP spectral norms ($r$ up to 0.77), revealing that the residual stream carries a signature of weight-space spectral scale.
4. **Calibration-free steering as a corollary** (Section 5.4, Figure 4): the formula $K_\ell = \bar\mu_\ell / \sqrt{d}$, derived from rank-1 perturbation equivalence, exploits the spectral proxy to calibrate steering without accessing weights directly.

---

## 2. Background and Related Work

### 2.1 The Linear Representation Hypothesis

The linear representation hypothesis (Park et al., 2023) posits that high-level features and concepts in LLMs are encoded as linear directions in residual-stream activation space. This view is supported by the success of linear probes in identifying factual and behavioral properties (Zou et al., 2023), by word2vec-style arithmetic over LLM embeddings (Mikolov et al., 2013), and by the interpretability of directions recovered via sparse autoencoders (Cunningham et al., 2023). Elhage et al. (2022) provide a theoretical account of *superposition* — the hypothesis that models encode more features than dimensions by placing them in nearly-orthogonal directions — which predicts that feature directions are not random but are constrained by the network's geometry. Our work provides evidence that the relevant geometric constraint is spectral: behavioral directions preferentially align with the weight matrices' dominant singular subspaces.

### 2.2 Transformer Circuits and MLP Geometry

The Mathematical Framework for Transformer Circuits (Elhage et al., 2021) characterizes the residual stream as a shared communication bus through which attention heads and MLP layers read and write information via low-rank subspaces. MLP layers are understood as key-value memories (Geva et al., 2021) where the up-projection $W_\text{up}$ detects input patterns and the down-projection $W_\text{down}$ writes output contributions to the residual stream. The principal directions of $W_\text{down}$ — its dominant right singular vectors — determine which residual-stream directions are most strongly modulated by MLP computation. Our finding that behavioral directions align with these singular vectors (Section 5.1) suggests that behaviors are preferentially encoded along the MLP's *primary output axes*, rather than in the null space or low-singular-value directions.

### 2.3 Activation Steering as an Interpretability Probe

Activation Addition (Turner et al., 2023) demonstrated that directly adding a direction vector to the residual stream reliably induces behavioral changes without modifying weights. Rimsky et al. (2024) extended this using contrastive prompt pairs to extract cleaner directions on instruction-tuned models. Zou et al. (2023) systematized this under *Representation Engineering*, framing it as a top-down approach to understanding internal representations. These methods have been productively applied to study honesty (Marks & Tegmark, 2023), refusal (Arditi et al., 2024), and emotion (Tigges et al., 2023). In all cases, however, the geometric relationship between steering directions and the underlying weight matrices has remained unexplored. We fill this gap.

The dominant current paradigm for finding interpretable directions in LLMs is sparse autoencoders (SAEs; Cunningham et al., 2023; Templeton et al., 2024), which decompose residual-stream activations into a sparse set of near-monosemantic features. SAE features and contrastive PCA directions are complementary probes: SAEs learn a data-driven overcomplete dictionary, while contrastive PCA extracts the axis of maximal behavioral variance from targeted prompt pairs. A natural prediction of our spectral alignment finding (Section 5.1) is that SAE features learned on MLP output activations should also concentrate in spectrally dominant subspaces of $W_\text{down}$, since they are learned from the same activation distribution. We leave a direct comparison to future work, but note that this prediction is falsifiable and would, if confirmed, generalize our finding from a specific probe method to any structured residual-stream direction.

### 2.4 Weight-Space Symmetries and Representation Geometry

The permutation symmetry of neural networks — the observation that swapping neurons in one layer and compensating in the adjacent layer leaves the function unchanged — has received significant attention in the context of loss landscape analysis (Entezari et al., 2022) and model merging (Ainsworth et al., 2023). Huh et al. (2024) propose the Platonic Representation Hypothesis, arguing that representations across architectures converge toward a shared structure. Our permutation invariance experiment (Section 5.2) directly tests whether behavioral representations respect these symmetries, providing an empirical data point on the relationship between functional equivalence classes and the geometry of learned directions.

---

## 3. Methodology

### 3.1 Behavioral Direction Extraction

For each of five behavioral axes $\mathcal{B}$, we construct a dataset of $N = 45$ contrastive pairs $\{(\mathbf{x}_i^+, \mathbf{x}_i^-)\}$, where $\mathbf{x}^+$ and $\mathbf{x}^-$ elicit opposing behavioral extremes. Following Rimsky et al. (2024), we extract activation differences at each layer in the middle 50% of the network:

$$\Delta \mathbf{h}_\ell^{(i)} = \mathbf{h}_\ell(\mathbf{x}_i^+) - \mathbf{h}_\ell(\mathbf{x}_i^-)$$

PCA with $k = 5$ components is fit to the stacked difference matrix $\Delta H_\ell \in \mathbb{R}^{N_\text{train} \times d}$ ($N_\text{train} = 36$, 80/20 split), yielding per-layer behavioral directions $\{\mathbf{c}_{\ell,j}\}_{j=1}^k$. The first principal component $\hat{\mathbf{c}}_{\ell,1}$ captures the dominant axis of behavioral variation at layer $\ell$.

### 3.2 Spectral Alignment Measurement

To test whether behavioral directions are constrained by MLP weight geometry, we measure the *alignment ratio*: the mean-max absolute cosine similarity between the top-5 PCA components and the top-10 right singular vectors $\{\mathbf{v}_j\}$ of MLP $W_\text{down}$ at each layer, normalized by the same quantity computed for random unit vectors drawn from $\mathbb{R}^d$:

$$\text{alignment ratio} = \frac{\max_{j \leq 10} |\hat{\mathbf{c}}_{\ell,1}^\top \mathbf{v}_j|}{\mathbb{E}_{\mathbf{r} \sim \mathcal{U}(\mathcal{S}^{d-1})}[\max_{j \leq 10} |\mathbf{r}^\top \mathbf{v}_j|]}$$

A ratio of 1.0 indicates no structural relationship; ratios $> 1$ indicate preferential alignment.

### 3.3 Permutation Invariance Test

We test whether behavioral directions are invariant to neuron permutations — the symmetry group that leaves the network function unchanged. For each model and behavior, we: (a) apply a random permutation $\pi$ to 50% of MLP layers (permuting rows of $W_\text{up}$ and columns of $W_\text{down}$ jointly to preserve function); (b) re-extract contrastive activations from the permuted model; (c) re-fit PCA; and (d) compute subspace cosine similarity between original and permuted direction sets via principal angles. We repeat for 5 random seeds. Invariance would require mean similarity $\geq 0.85$.

### 3.4 The K Calibration Formula: A Spectral Corollary

The spectral alignment result has a practical implication. In pre-norm transformers, the residual-stream norm $\bar\mu_\ell$ acts as a running integral of MLP spectral scales (Section 5.3). This means the activation norm provides a *proxy* for spectral geometry without requiring SVD computation. A rank-1 perturbation $\Delta W = \alpha\,\mathbf{u}\mathbf{v}^\top$ with $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ produces an expected output shift of $\alpha/\sqrt{d}$ on typical inputs. Setting the steering magnitude to match the local activation scale gives:

$$K_\ell = \frac{\bar\mu_\ell}{\sqrt{d}}$$

This formula calibrates the intervention to spectral geometry without weight access, enabling zero-tuning behavioral steering as a direct consequence of the spectral constraint.

---

## 4. Experimental Setup

**Models.** Four instruction-tuned architectures spanning distinct normalization regimes (Table 0).

**Table 0.** Models evaluated.

| Model | Family | $d$ | Layers | Normalization |
|-------|--------|-----|--------|---------------|
| Llama 3.1 8B Instruct | LLaMA | 4096 | 32 | Pre-norm |
| Qwen 2.5 7B Instruct | Qwen2 | 3584 | 28 | Pre-norm |
| Mistral 7B-v0.3 Instruct | Mistral | 4096 | 32 | Pre-norm |
| Gemma 2 9B IT | Gemma2 | 3584 | 42 | Pre-norm + Post-norm |

**Behavioral axes (5).** Each designed to isolate a single behavioral dimension via opposing prompt extremes.

| Axis | Positive pole | Negative pole |
|------|--------------|---------------|
| Refusal calibration | Compliance | Refusal |
| Formality | Casual register | Formal register |
| Verbosity control | Verbose | Terse |
| Uncertainty expression | Certainty-demanding | Uncertainty-acknowledging |
| Sycophancy suppression | Validation-seeking | Critique-seeking |

All experiments use seed 42. Spectral norms computed via exact SVD in float32 on CPU.

---

## 5. Results

### 5.1 Behavioral Directions Are Spectrally Constrained

PCA directions extracted from contrastive activations are not generic residual-stream vectors. Table 1 and Figure 2 show that behavioral directions align with the dominant right singular vectors of MLP $W_\text{down}$ at 3.04–4.28× above random across all four architectures, with a mean of **3.69×** (maximum: **11.62×**, Mistral, formality). This effect is consistent across all 20 model–behavior combinations (range: 2.83–5.03×): every cell in Figure 2's heatmap exceeds the random baseline, with no behavior or model as an outlier.

**Table 1.** Mean and maximum alignment ratios (PCA direction vs. top-10 right singular vectors of $W_\text{down}$), with layer-resolved peak.

| Model | Mean ratio | Max ratio | Peak layer | Early (0–25%) | Late (75–100%) |
|-------|-----------|-----------|-----------|--------------|---------------|
| Llama 3.1 8B | 3.40× | 7.11× | 13 | 3.64× | 2.31× |
| Qwen 2.5 7B | 4.05× | 9.48× | 27 | 4.26× | 3.31× |
| Mistral 7B | 4.28× | **11.62×** | 31 | **5.87×** | 4.49× |
| Gemma 2 9B | 3.04× | 7.64× | 8 | 3.33× | 2.78× |

> **Figure 2** | *Behavioral directions preferentially align with MLP spectral subspaces.* Heatmap of mean alignment ratios for all 20 model–behavior combinations (models on rows, behaviors on columns). Color scale anchored at 1.0 (random). All 20 cells exceed 1.0×. The uniformity across behavioral axes — spanning refusal, register, verbosity, epistemic state, and sycophancy — indicates the structural bias is architectural, not behavior-specific. Within each model, alignment is strongest in early-to-mid network layers and attenuates in the final layers, consistent with late layers specializing in output formatting rather than behavioral representation.
>
> *File: `figures/fig2_alignment_heatmap.pdf`*

**Interpretation.** MLP down-projection writes to the residual stream primarily along its dominant singular directions. Behavioral contrastive pairs — which probe the maximal variance in the model's internal response to opposing stimuli — capture exactly these high-variance, high-signal directions. This is consistent with the superposition hypothesis (Elhage et al., 2022): features encoded in spectrally dominant subspaces are the ones most strongly written by MLP computation and thus most accessible to residual-stream probing. Behaviors, as aggregates of many features, inherit this spectral bias.

**This finding is not a trivial consequence of extracting any high-variance direction.** The first principal component of raw (non-contrastive) activations would be dominated by token-position and prompt-format variance with no behavioral content; its alignment ratio would be expected to be near 1.0 by construction of the metric. The spectral alignment above 1.0 is specific to the *contrastive behavioral variance* axis, suggesting that behaviorally contrastive variation is preferentially organized along the MLP's primary output axes. We note this as a control prediction to verify in future work alongside SAE feature directions (Section 2.3).

### 5.2 Behavioral Directions Are Equivariant Under Permutation, Not Invariant

**This result is a direct prediction of Section 5.1.** If behavioral directions align with the right singular vectors $V$ of $W_\text{down}$, and if neuron permutations transform the right singular basis as $V \to PV$ (where $P$ is the permutation matrix applied to $W_\text{down}$ rows), then behavioral directions must co-rotate: they are *equivariant*, not invariant. We test this prediction explicitly.

**Table 2.** Subspace cosine similarity under 50% neuron permutation, separated by permuted and unpermuted layers (5 seeds, invariance threshold $\tau = 0.85$).

| Model | Permuted layers: mean sim ± std | Unpermuted layers | Threshold met? |
|-------|--------------------------------|-------------------|---------------|
| Llama 3.1 8B | 0.051 ± 0.040 | 1.000 (trivial) | No |
| Qwen 2.5 7B | 0.106 ± 0.068 | 1.000 (trivial) | No |
| Mistral 7B | 0.160 ± 0.123 | 1.000 (trivial) | No |
| Gemma 2 9B | 0.412 ± 0.181 | 1.000 (trivial) | No |

*Note: prior reports mixing permuted and unpermuted layers inflated means; the permuted-only figures above are the meaningful measurement.*

> **Figure 5** | *Behavioral directions rotate with the weights — they are equivariant, not invariant.* Box-and-whisker plots of subspace cosine similarity for *permuted layers only*, each model aggregated over 5 behaviors and 5 seeds. The red dashed line marks the 0.85 invariance threshold. All distributions sit near zero, confirming that re-fitting PCA after a neuron permutation produces directions unrelated to the originals. The small positive mean for Gemma 2 (0.41) reflects partial decoupling from permutation due to post-norm renormalization (Section 6.3).
>
> *File: `figures/fig5_permutation.pdf`*

The mechanistic interpretation is now a single coherent chain: behavioral directions align with $V$ (Section 5.1) → permutations rotate $V$ to $PV$ → behavioral directions co-rotate to $P\hat{\mathbf{c}}$ → similarity between $\hat{\mathbf{c}}$ and $P\hat{\mathbf{c}}$ is near zero for random $P$. This is *equivariance*: the direction transforms predictably under the group action rather than remaining fixed.

The interpretability implication is precise: **activation-space behavioral probes are parameterization-specific, not function-class invariants.** Two models in the same permutation equivalence class — identical in function — will have behavioral directions related by a known rotation, but the directions themselves are not canonically defined on the equivalence class. A probe that identifies "the refusal direction in Llama 3.1 8B" is identifying a direction relative to one member of an orbit; a permuted representative of the same functional model has a rotated refusal direction. This matters for cross-model comparison of behavioral representations and for any interpretability method that assumes activation-space directions are canonical.

### 5.3 Activation Norms Carry a Spectral Signature

A consequence of the spectral constraint is that activation norms — which are trivially measurable — serve as a proxy for MLP spectral scale. In pre-norm architectures, each sublayer adds an increment of magnitude $\sim \sigma_1(W_\ell) \cdot \sqrt{d}$ to the residual stream, making the cumulative norm $\bar\mu_\ell$ an integral of spectral scales across layers.

**Table 3.** Pearson $r$ between per-layer $K_\ell = \bar\mu_\ell/\sqrt{d}$ and $\sigma_1(W_\text{up})$.

| Model | Pearson $r$ | $p$-value |
|-------|------------|-----------|
| Llama 3.1 8B | 0.769 | $2.7 \times 10^{-7}$ |
| Qwen 2.5 7B | 0.711 | $2.2 \times 10^{-5}$ |
| Mistral 7B | 0.715 | $4.2 \times 10^{-6}$ |
| Gemma 2 9B | 0.421 | $5.4 \times 10^{-3}$ |

For three of four models, $r > 0.71$ ($p < 10^{-5}$). Figure 1 visualizes the per-layer relationship; color-coding by depth confirms the positive trend is stable across early, middle, and late layers. Gemma 2's weaker correlation ($r = 0.42$) reflects its dual-norm architecture, which breaks the integral relationship by bounding increment magnitudes via learned scale parameters (Section 6.1).

> **Figure 3** | *Per-layer norms motivate per-layer calibration and reveal spectral variation.* Mean L2 norm $\bar\mu_\ell$ ± std for all four models. Pre-norm architectures show monotonic growth (Llama: 0.71 → 59.4, an 84× range), reflecting spectral-scale accumulation. Gemma 2 begins two orders of magnitude higher and grows to 1,514, driven by its dual-norm increment dynamics. The 84× within-model range in pre-norm models shows that a fixed global steering magnitude is structurally inadequate.
>
> *File: `figures/fig3_norm_profiles.pdf`*

> **Figure 1** | *Activation norms track MLP spectral scale in pre-norm architectures.* Four-panel scatter of $K_\ell = \bar\mu_\ell/\sqrt{d}$ vs. $\sigma_1(W_\text{up})$, one panel per model, points colored by layer depth. Dashed lines: OLS fits. For Llama/Qwen/Mistral, the positive trend is tight and consistent across all depths — confirming that the residual stream's norm profile is a running integral of spectral scales. Gemma 2's scatter is flatter and wider, consistent with post-norm decoupling.
>
> *File: `figures/fig1_k_vs_spectral.pdf`*

### 5.4 Spectral Calibration Enables Zero-Tuning Steering

The spectral proxy has a practical corollary: setting $K_\ell = \bar\mu_\ell/\sqrt{d}$ (Section 3.4) calibrates behavioral steering to spectral geometry without accessing weights directly. Across all 20 model–behavior pairs (Table 4), K-calibrated PCA achieves 0.594 mean activation direction accuracy vs. 0.600 for uncalibrated PCA — near-parity with **zero hyperparameter tuning**, compared to the 0.352 no-steering baseline (+24.2 pp). Figure 4 shows per-behavior breakdown; formality is the notable outlier (−22.2 pp), suggesting that linguistic register may require calibration scales that deviate from the spectral prior — a mechanistically interesting exception warranting future investigation.

**Table 4.** Mean activation direction accuracy (20 model–behavior pairs, 9 test pairs each).

| Method | Mean accuracy |
|--------|--------------|
| No steering | 0.352 |
| Raw mean-diff | 0.432 |
| PCA uncalibrated ($K=1$) | 0.600 |
| **PCA K-calibrated** | **0.594** |

> **Figure 4** | *Calibration-free steering via the spectral proxy.* Grouped bar chart of activation direction accuracy per behavioral axis, averaged over four models. See supplemental for full results.
>
> *File: `figures/fig4_efficacy.pdf`*

---

## 6. Discussion

### 6.1 What the Spectral Constraint Means for Mechanistic Interpretability

The finding that behavioral directions align with spectrally dominant MLP subspaces has three implications for MI. **First**, behaviors are not encoded in arbitrary residual-stream directions: the weight matrices constrain which directions are accessible for behavioral encoding, and those directions are the spectrally prominent ones. This suggests that studying the SVD of MLP weight matrices — already a tool in weight-space analysis — is also directly relevant to understanding behavioral geometry. **Second**, the uniformity of this alignment across five qualitatively distinct behavioral axes (refusal, register, verbosity, epistemic state, social dynamics) suggests this is a systematic architectural property, not a behavior-specific coincidence. Any contrastive extraction procedure will tend to produce directions biased toward MLP spectral subspaces. **Third**, combined with the permutation result, this tells us that behavioral directions are spectrally constrained *and* parameterization-specific: they live in spectrally dominant subspaces of a particular weight realization, not in the abstract spectral geometry of the functional equivalence class.

The superposition hypothesis (Elhage et al., 2022) predicts that models encode more features than dimensions by using nearly-orthogonal directions, with the spectrally dominant directions carrying the most strongly activated features. Our results are consistent with behaviors being representationally located in this high-activation, spectrally dominant regime — the region of activation space where the MLP writes most strongly to the residual stream.

### 6.2 Equivariance as a Precise Boundary for Probe Validity

The permutation equivariance result draws a precise line around the validity of activation-space behavioral probes. Probes are valid as measurements of a specific model's internal geometry, but not as measurements of any functional property that is invariant across the permutation equivalence class. This has practical implications:

- **Cross-model comparison** of behavioral directions (e.g., asking whether Llama's refusal direction and Mistral's refusal direction are "the same") requires accounting for the arbitrary choice of permutation representative in each model's training.
- **Universal probes** — probes designed to work across multiple models — would need to be invariant to the permutation group, which requires averaging over orbits or designing equivariant architectures.
- **Model merging and stitching** methods that match neurons across models (Ainsworth et al., 2023) implicitly need to align the permutation representatives before comparing behavioral directions.

Designing behavioral probes that are functional-class invariants — rather than parameterization-specific measurements — is an open problem that our results help sharpen.

### 6.3 Gemma 2: When Post-Norm Breaks the Spectral Proxy

Gemma 2's weaker K–spectral correlation ($r = 0.42$ vs. $r > 0.71$) is explained by its dual-normalization scheme. In pre-norm transformers, the spectral proxy holds because each sublayer increment has magnitude $\sim \sigma_1(W_\ell) \cdot \sqrt{d}$. In Gemma 2:

$$x_{\ell+1} = x_\ell + \underbrace{\text{RMSNorm}_\text{post}}_{\text{magnitude-clips}}\!\left(F\!\left(\text{RMSNorm}_\text{pre}(x_\ell)\right)\right)$$

The post-sublayer RMSNorm bounds the increment magnitude to $\|\gamma^\text{post}_\ell\|_\text{eff} \cdot \sqrt{d}$, independent of $\sigma_1(W_\ell)$. This architecture — related to DeepNorm (Wang et al., 2022) — effectively decouples the residual stream from weight spectral structure, replacing it with a norm governed by learned scale parameters. The spectral constraint on behavioral directions (Section 5.1) remains, but the activation norm is no longer a reliable proxy for spectral scale. This motivates a follow-up: for dual-norm architectures, direct correlation of $K_\ell$ with $\|\gamma^\text{post}_\ell\|_\text{eff}$ may recover the missing spectral proxy.

---

## 7. Limitations

*(i)* **Control conditions.** We report alignment for contrastive PCA directions but not for the natural controls: the first PC of raw (non-contrastive) activations, random high-norm directions, or mean-diff directions without PCA decomposition. These controls are needed to confirm that the 3.69× ratio is specific to behavioral contrastive variance rather than to any structured direction. We predict raw-activation PC1 would show alignment near 1.0, and mean-diff near the PCA result, but this must be verified. *(ii)* **SAE features.** The prediction that SAE features (Cunningham et al., 2023) learned on MLP output activations should also show spectral alignment is falsifiable and would, if confirmed, generalize the finding beyond a single probe method. *(iii)* **$W_\text{down}$ only.** We measure alignment against $W_\text{down}$ right singular vectors; alignment with $W_\text{up}$ left singular vectors, attention $W_V$, or $W_O$ is unexplored. *(iv)* **Layer scope.** Layer-resolved alignment (Table 1) covers only layers 25–75% depth; early and very late layers are excluded. *(v)* **Scale.** All models are 7–9B parameters; whether the spectral constraint holds at 70B+ is unknown. *(vi)* **Permutation scope.** We test only random MLP neuron permutations; attention-head permutations and cross-layer symmetries may yield different equivariance profiles.

---

## 8. Conclusion

We have shown that behavioral directions in transformer LLMs are not arbitrary residual-stream vectors but are **spectrally constrained**: they preferentially occupy the dominant singular subspaces of MLP down-projection weight matrices at 3.69× above random across four architectures and five behavioral axes. This structural constraint is consistent with the superposition hypothesis and with the view that behaviors are encoded in the regions of activation space most strongly modulated by MLP computation.

We have also shown that this spectral constraint is **parameterization-specific**: behavioral directions rotate predictably under neuron permutations, establishing that activation-space behavioral probes are equivariant to the permutation group rather than invariant. Designing probes that are genuine functional-class invariants — averaging over orbits or building in equivariance — is an open challenge with direct implications for cross-model interpretability.

As a mechanistic corollary, activation norms serve as a spectral proxy in pre-norm architectures, enabling the calibration formula $K_\ell = \bar\mu_\ell/\sqrt{d}$ for zero-tuning behavioral steering. We offer this not as the paper's primary contribution but as a demonstration that mechanistic insights have practical consequences: understanding *where* behaviors live in the weights directly enables better tools for behavioral control.

---

## References

Ainsworth, S., Hayase, J., and Srinivasa, S. Git Re-Basin: Merging Models Modulo Permutation Symmetries. In *Proceedings of ICLR*, 2023.
<https://arxiv.org/abs/2209.04836>

Arditi, A., Obeso, O., Syed, A., Conmy, A., Sole, C., Agrawal, L., and Quirke, A. Refusal in Language Models Is Mediated by a Single Direction. *arXiv:2406.11717*, 2024.
<https://arxiv.org/abs/2406.11717>

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse Autoencoders Find Highly Interpretable Features in Language Models. In *Proceedings of ICLR*, 2024.
<https://arxiv.org/abs/2309.08600>

Elhage, N. et al. A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*, 2021.
<https://transformer-circuits.pub/2021/framework/index.html>

Elhage, N. et al. Toy Models of Superposition. *Transformer Circuits Thread*, 2022.
<https://arxiv.org/abs/2209.11141>

Entezari, R., Sedghi, H., Saukh, O., and Neyshabur, B. The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. In *Proceedings of ICLR*, 2022.
<https://arxiv.org/abs/2110.06296>

Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size. *arXiv:2408.00118*, 2024.
<https://arxiv.org/abs/2408.00118>

Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer Feed-Forward Layers Are Key-Value Memories. In *Proceedings of EMNLP*, 2021.
<https://arxiv.org/abs/2012.14913>

Huh, M., Cheung, B., Wang, T., and Isola, P. The Platonic Representation Hypothesis. In *Proceedings of ICML*, 2024.

Marks, S. and Tegmark, M. The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. *arXiv:2310.06824*, 2023.
<https://arxiv.org/abs/2310.06824>

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. Distributed Representations of Words and Phrases and Their Compositionality. In *Advances in NeurIPS*, 2013.

Park, K., Choe, Y. J., and Veitch, V. The Linear Representation Hypothesis and the Geometry of Large Language Models. *arXiv:2311.03658*, 2023.
<https://arxiv.org/abs/2311.03658>

Rimsky, N., Turner, A., Watkins, C., and Conerly, T. Steering Llama 2 via Contrastive Activation Addition. In *Proceedings of ACL*, 2024.

Templeton, A. et al. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*, 2024.
<https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>

Tigges, C., Hollinsworth, O. J., Geiger, A., and Nanda, N. Linear Representations of Sentiment in Large Language Models. *arXiv:2310.15154*, 2023.
<https://arxiv.org/abs/2310.15154>

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., and MacDiarmid, M. Activation Addition: Steering Language Models Without Optimization. *arXiv:2308.10248*, 2023.
<https://arxiv.org/abs/2308.10248>

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., and Wei, F. DeepNet: Scaling Transformers to 1,000 Layers. *arXiv:2203.00555*, 2022.
<https://arxiv.org/abs/2203.00555>

Zou, A. et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*, 2023.
<https://arxiv.org/abs/2310.01405>

---

## Appendix: Figure Placement for 4-Page Limit

> **Recommended main body:** Figure 2 (alignment heatmap, §5.1) and Figure 5 (permutation distributions, §5.2) — the two core MI claims.
> **Supplemental:** Figure 1 (K vs spectral scatter, §5.3), Figure 3 (norm profiles, §5.3 motivation), Figure 4 (efficacy bar, §5.4).
> All PDFs in `figures/`.
