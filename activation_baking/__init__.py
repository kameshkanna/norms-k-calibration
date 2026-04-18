"""
activation_baking: PCA-directed activation adapters via weight-space symmetry calibration.

Research framework for ICML 2026 Workshop on Weight-Space Symmetries.

This package provides an end-to-end pipeline for:
    * Extracting contrastive activation differences from paired prompts.
    * Fitting PCA-based behavioral directions to the resulting diffs.
    * Calibrating steering magnitudes (K values) from per-layer activation norms.
    * Applying calibrated steering via forward hooks during generation.
    * Evaluating behavioral shift, KL divergence, and permutation invariance.

Typical usage::

    from activation_baking import Baker, BehavioralEvaluator

    baker = Baker("meta-llama/Llama-3-8B-Instruct")
    baker.fit(positive_prompts, negative_prompts)
    steered = baker.generate(test_prompts, alpha=1.5)

    evaluator = BehavioralEvaluator()
    result = evaluator.evaluate(baker, positive_test, negative_test, "helpfulness")
"""

from activation_baking.baker import Baker
from activation_baking.calibrator import KCalibrator
from activation_baking.evaluator import BehavioralEvaluator, EvaluationResult
from activation_baking.extractor import ActivationExtractor
from activation_baking.model_utils import ModelInfo
from activation_baking.pca_director import BehavioralDirections, PCADirector

__all__ = [
    "Baker",
    "BehavioralDirections",
    "BehavioralEvaluator",
    "EvaluationResult",
    "KCalibrator",
    "ActivationExtractor",
    "ModelInfo",
    "PCADirector",
]

__version__ = "0.1.0"
__author__ = "Activation Baking Research Team"
