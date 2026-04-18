"""
BehavioralEvaluator: Quantitative evaluation of activation-steering quality.

Measures behavioral shift (cosine similarity in activation space), KL divergence
between baseline and steered token distributions, and subspace similarity between
two sets of PCA directions (used for the permutation invariance experiment).
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from activation_baking.baker import Baker

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Aggregate evaluation metrics for a single steering experiment.

    Attributes:
        behavior_name: Human-readable label for the target behaviour.
        model_id: HuggingFace model identifier.
        method: Steering method used; one of ``"none"``, ``"raw_addition"``,
            ``"pca_uncalibrated"``, ``"pca_k_calibrated"``.
        alpha: Alpha multiplier applied during steering.
        baseline_similarity: Mean cosine similarity of baseline response
            activations to positive-example activations.
        steered_similarity: Mean cosine similarity of steered response
            activations to positive-example activations.
        behavioral_shift: ``steered_similarity - baseline_similarity``.
            Positive values indicate movement toward the target behaviour.
        kl_divergence: Mean KL(baseline || steered) over the test prompts.
        n_test_pairs: Number of (positive, negative) test pairs evaluated.
    """

    behavior_name: str
    model_id: str
    method: str
    alpha: float
    baseline_similarity: float
    steered_similarity: float
    behavioral_shift: float
    kl_divergence: float
    n_test_pairs: int


class BehavioralEvaluator:
    """
    Evaluates the quality of PCA-directed activation steering.

    All heavy computation is delegated to the Baker's internals; this class
    only orchestrates extraction, comparison, and metric computation.

    Methods
    -------
    evaluate : Full end-to-end evaluation producing an :class:`EvaluationResult`.
    compute_subspace_similarity : Principal-angle similarity for two direction sets.
    compute_kl_divergence : Batch-averaged KL divergence between logit distributions.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        baker: "Baker",
        positive_test: List[str],
        negative_test: List[str],
        behavior_name: str,
        method: str = "pca_k_calibrated",
        alpha: float = 1.0,
    ) -> EvaluationResult:
        """
        Evaluate behavioral steering quality on held-out test prompts.

        Procedure:
        1. Generate baseline responses (no hooks).
        2. Generate steered responses.
        3. Extract last-token residual-stream activations for both.
        4. Compute cosine similarities to positive-example activations.
        5. Compute KL divergence between baseline and steered logit distributions.

        Parameters
        ----------
        baker:
            A fitted :class:`~activation_baking.baker.Baker` instance.
        positive_test:
            Held-out positive (target-behaviour) prompts.
        negative_test:
            Held-out negative (null-behaviour) prompts.  Same length as
            `positive_test`.
        behavior_name:
            Human-readable label stored in the returned result.
        method:
            Method tag stored in the returned result (for downstream analysis).
        alpha:
            Alpha multiplier forwarded to :py:meth:`Baker.generate`.

        Returns
        -------
        EvaluationResult

        Raises
        ------
        RuntimeError
            If `baker` is not fitted.
        ValueError
            If `positive_test` and `negative_test` differ in length, or are empty.
        TypeError
            If inputs have incorrect types.
        """
        if not isinstance(positive_test, list) or not isinstance(negative_test, list):
            raise TypeError("positive_test and negative_test must be lists of strings.")
        if len(positive_test) != len(negative_test):
            raise ValueError(
                f"positive_test ({len(positive_test)}) and negative_test "
                f"({len(negative_test)}) must have equal length."
            )
        if not positive_test:
            raise ValueError("positive_test must not be empty.")
        if any(not isinstance(p, str) for p in positive_test + negative_test):
            raise TypeError("All test prompts must be strings.")
        if method not in {"none", "raw_addition", "pca_uncalibrated", "pca_k_calibrated"}:
            raise ValueError(
                f"method must be one of 'none', 'raw_addition', 'pca_uncalibrated', "
                f"'pca_k_calibrated'; got '{method}'."
            )
        if not hasattr(baker, "_is_fitted") or not baker._is_fitted:
            raise RuntimeError("Baker is not fitted; call baker.fit() first.")

        n_test_pairs = len(positive_test)
        fitted_layers = baker.fitted_layers

        # --- 1. Generate responses ---
        self._logger.info(
            "Generating baseline responses for %d prompts...", n_test_pairs
        )
        baseline_responses: List[str] = baker.generate_baseline(
            prompts=positive_test, max_new_tokens=100
        )

        self._logger.info(
            "Generating steered responses (alpha=%.4f)...", alpha
        )
        steered_responses: List[str] = baker.generate(
            prompts=positive_test, alpha=alpha, max_new_tokens=100
        )

        # --- 2. Extract activations (last token, first fitted layer) ---
        # Use the median fitted layer as a representative measurement layer.
        mid_layer = fitted_layers[len(fitted_layers) // 2]

        self._logger.info(
            "Extracting activations at layer '%s' for similarity computation...",
            mid_layer,
        )
        # Reference: positive test prompts
        positive_acts: Dict[str, torch.Tensor] = baker._extractor.extract(
            prompts=positive_test,
            layer_names=[mid_layer],
            position="last",
        )
        positive_ref: torch.Tensor = positive_acts[mid_layer]  # [n, hidden]

        # Baseline response activations
        baseline_acts: Dict[str, torch.Tensor] = baker._extractor.extract(
            prompts=baseline_responses,
            layer_names=[mid_layer],
            position="last",
        )
        baseline_ref: torch.Tensor = baseline_acts[mid_layer]  # [n, hidden]

        # Steered response activations
        steered_acts: Dict[str, torch.Tensor] = baker._extractor.extract(
            prompts=steered_responses,
            layer_names=[mid_layer],
            position="last",
        )
        steered_ref: torch.Tensor = steered_acts[mid_layer]  # [n, hidden]

        # --- 3. Cosine similarities ---
        # Vectorised: [n] cosine similarities → scalar mean
        baseline_sim: float = self._mean_cosine_similarity(
            baseline_ref, positive_ref
        )
        steered_sim: float = self._mean_cosine_similarity(
            steered_ref, positive_ref
        )
        behavioral_shift: float = steered_sim - baseline_sim

        # --- 4. KL divergence ---
        self._logger.info("Computing KL divergence on test prompts...")
        kl_div: float = self._compute_kl_on_prompts(
            baker=baker, prompts=positive_test, alpha=alpha
        )

        gc.collect()
        device = next(baker._model.parameters()).device
        if device.type == "cuda":
            torch.cuda.empty_cache()

        result = EvaluationResult(
            behavior_name=behavior_name,
            model_id=baker._model_id,
            method=method,
            alpha=alpha,
            baseline_similarity=baseline_sim,
            steered_similarity=steered_sim,
            behavioral_shift=behavioral_shift,
            kl_divergence=kl_div,
            n_test_pairs=n_test_pairs,
        )
        self._logger.info(
            "Evaluation complete: behavioral_shift=%.4f, kl_divergence=%.4f",
            behavioral_shift,
            kl_div,
        )
        return result

    def compute_subspace_similarity(
        self,
        dirs_a: Dict[str, torch.Tensor],
        dirs_b: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute principal-angle cosine similarity between two PCA direction sets.

        This is the primary metric for the permutation invariance experiment: if
        K-calibrated PCA directions are truly invariant to neuron-permutation
        symmetry, scores should approach 1.0.

        Parameters
        ----------
        dirs_a:
            Mapping from layer name to component tensor ``[n_components, hidden_size]``.
            Typically the original (un-permuted) model's directions.
        dirs_b:
            Same structure for the permuted model.

        Returns
        -------
        Dict[str, float]
            Mean cosine of principal angles per layer (0.0 – 1.0).

        Raises
        ------
        TypeError
            If values in `dirs_a` or `dirs_b` are not torch.Tensor.
        ValueError
            If tensor shapes are inconsistent.
        """
        if not isinstance(dirs_a, dict) or not isinstance(dirs_b, dict):
            raise TypeError("dirs_a and dirs_b must be dicts.")

        scores: Dict[str, float] = {}
        shared_layers = set(dirs_a.keys()) & set(dirs_b.keys())
        missing = (set(dirs_a.keys()) | set(dirs_b.keys())) - shared_layers
        if missing:
            self._logger.warning(
                "compute_subspace_similarity: %d layers not shared between dirs_a "
                "and dirs_b; they will be skipped: %s",
                len(missing),
                missing,
            )

        for layer_name in shared_layers:
            comp_a = dirs_a[layer_name]
            comp_b = dirs_b[layer_name]

            if not isinstance(comp_a, torch.Tensor) or not isinstance(comp_b, torch.Tensor):
                raise TypeError(
                    f"Layer '{layer_name}': expected torch.Tensor, got "
                    f"{type(comp_a)} and {type(comp_b)}."
                )
            if comp_a.ndim != 2 or comp_b.ndim != 2:
                raise ValueError(
                    f"Layer '{layer_name}': tensors must be 2-D [n_components, hidden_size]; "
                    f"got shapes {tuple(comp_a.shape)} and {tuple(comp_b.shape)}."
                )
            if comp_a.shape[1] != comp_b.shape[1]:
                raise ValueError(
                    f"Layer '{layer_name}': hidden_size mismatch: "
                    f"{comp_a.shape[1]} vs {comp_b.shape[1]}."
                )

            # Principal angles via SVD of cross-Gram matrix
            cross_gram: torch.Tensor = comp_a.float() @ comp_b.float().T  # [k_a, k_b]
            singular_values = torch.linalg.svdvals(cross_gram)
            singular_values = singular_values.clamp(0.0, 1.0)
            scores[layer_name] = singular_values.mean().item()

        return scores

    def compute_kl_divergence(
        self,
        logits_baseline: torch.Tensor,
        logits_steered: torch.Tensor,
    ) -> float:
        """
        Compute mean KL divergence between baseline and steered token distributions.

        KL(baseline || steered) = Σ_v P_baseline(v) * log(P_baseline(v) / P_steered(v))

        Parameters
        ----------
        logits_baseline:
            Raw unnormalised logits from the baseline model, shape
            ``[batch, vocab_size]``.
        logits_steered:
            Raw unnormalised logits from the steered model, same shape.

        Returns
        -------
        float
            Batch-averaged KL divergence (nats).

        Raises
        ------
        TypeError
            If inputs are not torch.Tensor.
        ValueError
            If tensor shapes do not match, or are not 2-D.
        """
        if not isinstance(logits_baseline, torch.Tensor) or not isinstance(
            logits_steered, torch.Tensor
        ):
            raise TypeError(
                "logits_baseline and logits_steered must be torch.Tensor."
            )
        if logits_baseline.shape != logits_steered.shape:
            raise ValueError(
                f"Shape mismatch: logits_baseline {tuple(logits_baseline.shape)} vs "
                f"logits_steered {tuple(logits_steered.shape)}."
            )
        if logits_baseline.ndim != 2:
            raise ValueError(
                f"Expected 2-D tensors [batch, vocab_size], got "
                f"shape {tuple(logits_baseline.shape)}."
            )

        # Compute log-softmax on both; use float32 for numerical stability
        log_p = F.log_softmax(logits_baseline.float(), dim=-1)   # log P_baseline
        log_q = F.log_softmax(logits_steered.float(), dim=-1)    # log P_steered

        # KL(P || Q) per sample via F.kl_div which expects log-input
        # F.kl_div(log_q, log_p, ...) = Σ exp(log_p) * (log_p - log_q) = KL(P||Q)
        kl_per_sample: torch.Tensor = F.kl_div(
            input=log_q,
            target=log_p,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)  # [batch]

        return kl_per_sample.mean().item()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """
        Compute the mean pairwise cosine similarity between two activation matrices.

        Parameters
        ----------
        a:
            Activations, shape ``[n, hidden_size]``.
        b:
            Activations, shape ``[n, hidden_size]`` — matched row-by-row with `a`.

        Returns
        -------
        float
            Mean cosine similarity across the batch.

        Raises
        ------
        ValueError
            If `a` and `b` have different shapes.
        """
        if a.shape != b.shape:
            raise ValueError(
                f"_mean_cosine_similarity: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}."
            )
        a_norm = F.normalize(a.float(), p=2, dim=-1)
        b_norm = F.normalize(b.float(), p=2, dim=-1)
        # Element-wise dot product → [n] → scalar
        cosine_sims: torch.Tensor = (a_norm * b_norm).sum(dim=-1)
        return cosine_sims.mean().item()

    def _compute_kl_on_prompts(
        self,
        baker: "Baker",
        prompts: List[str],
        alpha: float,
    ) -> float:
        """
        Compute KL divergence between baseline and steered next-token distributions.

        Runs two forward passes (baseline + steered) and compares the last-token
        logits, which represent the next-token predictive distribution.

        Parameters
        ----------
        baker:
            A fitted Baker instance.
        prompts:
            List of input strings.
        alpha:
            Steering magnitude multiplier.

        Returns
        -------
        float
            Mean KL(baseline || steered) across prompts.
        """
        device = next(baker._model.parameters()).device

        inputs = baker._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # --- Baseline forward pass ---
        with torch.inference_mode():
            baseline_out = baker._model(**inputs)
        baseline_logits: torch.Tensor = baseline_out.logits[:, -1, :]  # [batch, vocab]

        # --- Steered forward pass (hooks active) ---
        hooks = baker._register_steering_hooks(alpha=alpha)
        try:
            with torch.inference_mode():
                steered_out = baker._model(**inputs)
            steered_logits: torch.Tensor = steered_out.logits[:, -1, :]
        finally:
            for h in hooks:
                h.remove()

        kl = self.compute_kl_divergence(
            logits_baseline=baseline_logits.cpu(),
            logits_steered=steered_logits.cpu(),
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return kl
