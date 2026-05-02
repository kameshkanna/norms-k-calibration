"""
Baker: End-to-end user-facing API for PCA-directed activation steering.

Loads a causal language model, fits behavioral directions from contrastive
prompt pairs, registers calibrated steering hooks during generation, and
provides save/load for the steering artefacts (not model weights).
"""

import copy
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from activation_baking.model_utils import ModelInfo, detect_model_info, get_layer_module
from activation_baking.extractor import ActivationExtractor
from activation_baking.calibrator import KCalibrator
from activation_baking.pca_director import BehavioralDirections, PCADirector

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    """
    Resolve the target device string to a concrete ``torch.device``.

    Parameters
    ----------
    device:
        ``"auto"`` selects the first available CUDA GPU, then MPS, then CPU.
        Any other string is passed directly to ``torch.device()``.

    Returns
    -------
    torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
        logger.debug("Device 'auto' resolved to '%s'.", resolved)
        return resolved
    return torch.device(device)


class Baker:
    """
    High-level API for fitting and applying PCA-directed activation steering.

    Workflow::

        baker = Baker("meta-llama/Llama-3-8B-Instruct")
        baker.fit(positive_prompts, negative_prompts)
        steered_responses = baker.generate(test_prompts, alpha=1.5)

    The class manages:
    * Model + tokenizer loading (HuggingFace Hub).
    * Layer-selective activation extraction via :class:`ActivationExtractor`.
    * K-value calibration via :class:`KCalibrator`.
    * PCA direction fitting via :class:`PCADirector`.
    * Forward-hook-based residual-stream steering during generation.
    * Serialisation of steering artefacts (model weights are *not* saved).

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (e.g. ``"meta-llama/Llama-3-8B-Instruct"``).
    device:
        Target device string.  ``"auto"`` picks CUDA → MPS → CPU.
    load_in_8bit:
        Enable bitsandbytes INT8 quantisation.
    load_in_4bit:
        Enable bitsandbytes NF4 quantisation.
    torch_dtype:
        Override model weight dtype (e.g. ``torch.bfloat16`` for H100).
        Defaults to ``float16`` on GPU and ``float32`` on CPU.
    attn_implementation:
        Attention backend passed to ``from_pretrained``
        (e.g. ``"flash_attention_2"`` for Flash Attention 2).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got: {model_id!r}")
        if load_in_8bit and load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive.")

        self._model_id: str = model_id
        self._device: torch.device = _resolve_device(device)
        self._device_str: str = device

        logger.info("Loading model '%s' onto device '%s'.", model_id, self._device)
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        _default_dtype = torch.float16 if self._device.type != "cpu" else torch.float32
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype if torch_dtype is not None else _default_dtype,
            "device_map": str(self._device),
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id, **load_kwargs
        )
        self._model.eval()

        self._model_info: ModelInfo = detect_model_info(self._model, model_id)
        self._extractor: ActivationExtractor = ActivationExtractor(
            model=self._model,
            tokenizer=self._tokenizer,
            model_info=self._model_info,
            device=str(self._device),
            batch_size=4,
        )
        self._calibrator: KCalibrator = KCalibrator()
        self._director: PCADirector = PCADirector()

        self._directions: Dict[str, BehavioralDirections] = {}
        self._k_values: Dict[str, float] = {}
        self._fitted_layers: List[str] = []
        self._is_fitted: bool = False

        logger.info("Baker initialised for model '%s'.", model_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def directions(self) -> Dict[str, BehavioralDirections]:
        """Fitted PCA directions per layer. Empty until :py:meth:`fit` is called."""
        return self._directions

    @property
    def k_values(self) -> Dict[str, float]:
        """Calibrated K values per layer. Empty until :py:meth:`fit` is called."""
        return self._k_values

    @property
    def fitted_layers(self) -> List[str]:
        """Layer names used during the most recent :py:meth:`fit` call."""
        return list(self._fitted_layers)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layers: Optional[Tuple[int, int]] = None,
        n_components: int = 5,
        k_calibration: Union[str, float] = "auto",
        n_norm_prompts: int = 50,
        use_mean_diff: bool = False,
    ) -> None:
        """
        Fit behavioral directions from contrastive prompt pairs.

        Steps:
        1. Select target layers (default: middle 50 % of the model's layers).
        2. Extract contrastive activation differences at each layer.
        3. Fit directions (PCA by default, or mean diff when use_mean_diff=True).
        4. Calibrate K values (via :class:`KCalibrator`).
        5. Attach K values to the fitted directions.

        Parameters
        ----------
        positive_prompts:
            Prompts that exemplify the *target* behaviour.
        negative_prompts:
            Prompts that exemplify the *opposite* / null behaviour.
            Must have the same length as `positive_prompts`.
        layers:
            ``(start_layer_idx, end_layer_idx)`` inclusive.  If ``None``,
            the middle 50 % of transformer layers is selected automatically.
        n_components:
            Number of PCA components to retain per layer. Ignored when
            ``use_mean_diff=True``.
        k_calibration:
            * ``"auto"``  — K = μ_norm / √hidden_size (recommended).
            * ``"none"``  — K = 1.0 for all layers.
            * ``float``   — constant K applied to every layer.
        n_norm_prompts:
            Number of prompts (drawn from `positive_prompts`) used to estimate
            the mean activation norm for K calibration.
        use_mean_diff:
            If ``True``, skip PCA and use the L2-normalised mean contrastive
            difference vector as the single steering direction.  This is the
            ``raw_addition`` ablation baseline: no subspace decomposition,
            just the classic steering-vector approach (Turner et al. 2023).
            The hook still runs through ``Baker.generate`` so generation is
            real, not a geometric estimate.

        Raises
        ------
        ValueError
            If `positive_prompts` and `negative_prompts` have different lengths,
            or if `layers` indices are out of range.
        TypeError
            If prompt lists contain non-string elements.
        """
        # --- Input validation ---
        if not isinstance(positive_prompts, list) or not isinstance(negative_prompts, list):
            raise TypeError("positive_prompts and negative_prompts must be lists.")
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError(
                f"positive_prompts ({len(positive_prompts)}) and "
                f"negative_prompts ({len(negative_prompts)}) must have equal length."
            )
        if any(not isinstance(p, str) for p in positive_prompts + negative_prompts):
            raise TypeError("All prompts must be strings.")
        if not isinstance(k_calibration, (str, int, float)):
            raise TypeError(
                f"k_calibration must be 'auto', 'none', or a float; got {type(k_calibration)}"
            )
        if isinstance(k_calibration, str) and k_calibration not in {"auto", "none"}:
            raise ValueError(
                f"k_calibration string must be 'auto' or 'none', got '{k_calibration}'"
            )
        if not isinstance(use_mean_diff, bool):
            raise TypeError(f"use_mean_diff must be bool, got {type(use_mean_diff)}")

        # --- Layer selection ---
        n_layers = self._model_info.num_layers
        if layers is None:
            quarter = n_layers // 4
            start_idx = quarter
            end_idx = n_layers - quarter - 1
        else:
            start_idx, end_idx = layers
            if not (0 <= start_idx <= end_idx < n_layers):
                raise ValueError(
                    f"layers=({start_idx}, {end_idx}) out of range for model with "
                    f"{n_layers} layers (valid: 0..{n_layers - 1})."
                )

        target_layer_names: List[str] = self._model_info.layer_module_names[
            start_idx : end_idx + 1
        ]
        logger.info(
            "Fitting on layers [%d..%d] (%d layers): %s ... %s",
            start_idx,
            end_idx,
            len(target_layer_names),
            target_layer_names[0],
            target_layer_names[-1],
        )

        # --- Extract contrastive diffs ---
        logger.info(
            "Extracting contrastive diffs for %d pairs...", len(positive_prompts)
        )
        activation_diffs: Dict[str, torch.Tensor] = (
            self._extractor.extract_contrastive_diffs(
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                layer_names=target_layer_names,
            )
        )
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Fit directions (PCA or mean-diff) ---
        if use_mean_diff:
            # Raw-addition ablation: use the L2-normalised mean contrastive diff
            # as a single steering direction, bypassing PCA entirely.
            # This replicates the classic Turner et al. (2023) approach and allows
            # fair comparison inside the same hook-based generation pipeline.
            logger.info("Fitting mean-diff direction (raw_addition ablation)...")
            directions = self._fit_mean_diff_directions(
                activation_diffs=activation_diffs,
                layer_names=target_layer_names,
            )
        else:
            logger.info("Fitting PCA (n_components=%d)...", n_components)
            directions = self._director.fit(
                activation_diffs=activation_diffs,
                n_components=n_components,
            )
        gc.collect()

        # --- K calibration ---
        k_values: Dict[str, float]
        if k_calibration == "none":
            k_values = {layer: 1.0 for layer in target_layer_names}
        elif isinstance(k_calibration, (int, float)):
            k_values = {layer: float(k_calibration) for layer in target_layer_names}
        else:  # "auto"
            norm_prompts = positive_prompts[:n_norm_prompts]
            logger.info(
                "Computing layer norms on %d prompts for K calibration...",
                len(norm_prompts),
            )
            layer_norms: Dict[str, float] = self._extractor.compute_layer_norms(
                prompts=norm_prompts,
                layer_names=target_layer_names,
            )
            k_values = self._calibrator.calibrate_all_layers(
                layer_norms=layer_norms,
                hidden_size=self._model_info.hidden_size,
            )
            gc.collect()
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        # --- Attach K values ---
        self._director.set_k_values(directions=directions, k_values=k_values)

        self._directions = directions
        self._k_values = k_values
        self._fitted_layers = list(directions.keys())
        self._is_fitted = True

        logger.info(
            "Baker.fit complete: %d layers fitted, k range=[%.4f, %.4f].",
            len(self._directions),
            min(k_values.values()),
            max(k_values.values()),
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Union[str, List[str]],
        alpha: float = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        **generation_kwargs: Any,
    ) -> List[str]:
        """
        Generate text with PCA-directed activation steering applied.

        Steering hooks are registered on each fitted layer before generation
        and unconditionally removed afterwards, even on exception.

        Parameters
        ----------
        prompts:
            Single string or list of strings.
        alpha:
            Global multiplier on the steering vector magnitude.
        max_new_tokens:
            Maximum number of new tokens to generate per prompt.
        temperature:
            Sampling temperature (passed to ``model.generate``).
        **generation_kwargs:
            Additional keyword arguments forwarded to ``model.generate``.

        Returns
        -------
        List[str]
            Decoded generated text for each input prompt (input prefix excluded).

        Raises
        ------
        RuntimeError
            If :py:meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Baker has not been fitted. Call baker.fit() before generate()."
            )
        return self._generate_impl(
            prompts=prompts,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steer=True,
            **generation_kwargs,
        )

    def generate_baseline(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        **generation_kwargs: Any,
    ) -> List[str]:
        """
        Generate text without any steering (baseline).

        Parameters
        ----------
        prompts:
            Single string or list of strings.
        max_new_tokens:
            Maximum number of new tokens to generate per prompt.
        temperature:
            Sampling temperature.
        **generation_kwargs:
            Additional keyword arguments forwarded to ``model.generate``.

        Returns
        -------
        List[str]
            Decoded generated text (input prefix excluded).
        """
        return self._generate_impl(
            prompts=prompts,
            alpha=1.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steer=False,
            **generation_kwargs,
        )

    def _generate_impl(
        self,
        prompts: Union[str, List[str]],
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        steer: bool,
        **generation_kwargs: Any,
    ) -> List[str]:
        """
        Shared generation implementation.

        Registers forward hooks on fitted layers when `steer=True`.
        Hooks are always removed in the ``finally`` block.

        Parameters
        ----------
        prompts:
            Single string or list of strings.
        alpha:
            Steering magnitude multiplier (ignored when `steer=False`).
        max_new_tokens:
            Maximum new tokens.
        temperature:
            Sampling temperature.
        steer:
            If True, register activation-steering hooks before generation.
        **generation_kwargs:
            Forwarded to ``model.generate``.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("prompts must be a non-empty list.")

        # Decoder-only models require left-padding during generation so that
        # the last real token aligns at position -1 for every sequence.
        orig_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._device)
        finally:
            self._tokenizer.padding_side = orig_padding_side

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        try:
            if steer:
                hooks = self._register_steering_hooks(alpha=alpha)

            with torch.inference_mode():
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0.0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    **generation_kwargs,
                }
                if temperature > 0.0:
                    gen_kwargs["temperature"] = temperature

                output_ids: torch.Tensor = self._model.generate(
                    **inputs, **gen_kwargs
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[:, input_len:]
            results: List[str] = self._tokenizer.batch_decode(
                new_ids, skip_special_tokens=True
            )
        finally:
            for h in hooks:
                h.remove()
            hooks.clear()

        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_steering_hooks(
        self, alpha: float
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register forward hooks on all fitted layers.

        The hook intercepts the layer output tuple, applies
        :py:meth:`PCADirector.apply_steering` to the residual stream
        (``output[0]``), and returns the modified tuple.

        Parameters
        ----------
        alpha:
            Steering magnitude multiplier passed to ``apply_steering``.

        Returns
        -------
        List[torch.utils.hooks.RemovableHandle]
            Handles that must be removed after generation.
        """
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        for layer_name, bd in self._directions.items():
            module = self._get_module_by_name(layer_name)
            if module is None:
                logger.warning(
                    "_register_steering_hooks: module '%s' not found; skipping.",
                    layer_name,
                )
                continue

            # Capture bd and alpha by value (closure per layer)
            def _make_hook(
                directions: BehavioralDirections, _alpha: float
            ):
                def hook(
                    module: nn.Module,
                    input: Any,  # noqa: A002
                    output: Any,
                ) -> Any:
                    """
                    Forward hook: steer residual-stream output of a decoder block.

                    Decoder blocks typically return a tuple where index 0 is the
                    residual stream tensor of shape [batch, seq_len, hidden_size].
                    We apply steering to every token position.
                    """
                    if isinstance(output, tuple):
                        residual = output[0]  # [batch, seq_len, hidden_size]
                        batch, seq_len, hidden = residual.shape
                        # Flatten to [batch*seq_len, hidden] for vectorised steering
                        flat = residual.reshape(batch * seq_len, hidden)
                        steered_flat = self._director.apply_steering(
                            activations=flat,
                            directions=directions,
                            alpha=_alpha,
                        )
                        steered = steered_flat.reshape(batch, seq_len, hidden)
                        return (steered,) + output[1:]
                    elif isinstance(output, torch.Tensor):
                        # Some architectures return a plain tensor
                        batch, seq_len, hidden = output.shape
                        flat = output.reshape(batch * seq_len, hidden)
                        steered_flat = self._director.apply_steering(
                            activations=flat,
                            directions=directions,
                            alpha=_alpha,
                        )
                        return steered_flat.reshape(batch, seq_len, hidden)
                    else:
                        logger.warning(
                            "Steering hook on '%s': unexpected output type %s; "
                            "skipping intervention.",
                            directions.layer_name,
                            type(output),
                        )
                        return output

                return hook

            handle = module.register_forward_hook(_make_hook(bd, alpha))
            hooks.append(handle)

        logger.debug("Registered %d steering hooks (alpha=%.4f).", len(hooks), alpha)
        return hooks

    def _get_module_by_name(self, module_name: str) -> Optional[nn.Module]:
        """
        Retrieve a submodule from ``self._model`` by its dot-separated path.

        Parameters
        ----------
        module_name:
            Dot-separated attribute path (e.g. ``"model.layers.16"``).

        Returns
        -------
        Optional[nn.Module]
            The resolved submodule, or ``None`` if the path does not exist.
        """
        parts = module_name.split(".")
        module: Any = self._model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try integer indexing for Sequential/ModuleList
                try:
                    idx = int(part)
                    module = module[idx]
                except (ValueError, TypeError, IndexError, KeyError):
                    return None
        return module if isinstance(module, nn.Module) else None

    def _fit_mean_diff_directions(
        self,
        activation_diffs: Dict[str, torch.Tensor],
        layer_names: List[str],
    ) -> Dict[str, "BehavioralDirections"]:
        """
        Build BehavioralDirections using the L2-normalised mean diff as the sole component.

        This is the raw_addition ablation: no PCA, just the classic mean contrastive
        diff vector (Turner et al. 2023).  It runs through the same hook-based
        generation pipeline so results are directly comparable to PCA methods.

        Parameters
        ----------
        activation_diffs:
            Layer name -> [n_pairs, hidden_size] float tensor of diffs.
        layer_names:
            Ordered list of layer names to process.

        Returns
        -------
        Dict[str, BehavioralDirections]
            One BehavioralDirections per layer with ``n_components=1`` and
            ``components`` = the unit-normalised mean diff.
        """
        import numpy as np
        directions: Dict[str, BehavioralDirections] = {}

        for layer_name in layer_names:
            diffs = activation_diffs[layer_name].float()   # [n_pairs, hidden]
            mean_diff = diffs.mean(dim=0)                  # [hidden]
            norm = torch.linalg.vector_norm(mean_diff)
            if norm < 1e-8:
                # Degenerate case: diffs cancel out; use zero vector
                unit_diff = mean_diff
            else:
                unit_diff = mean_diff / norm

            directions[layer_name] = BehavioralDirections(
                layer_name=layer_name,
                components=unit_diff.unsqueeze(0).cpu(),   # [1, hidden]
                explained_variance_ratio=np.array([1.0]),
                mean_diff=mean_diff.cpu(),
                n_pairs_fit=diffs.shape[0],
                k_value=None,
            )

        logger.info(
            "Mean-diff directions fitted for %d layers.", len(directions)
        )
        return directions

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        private: bool = False,
    ) -> None:
        """
        Save steering artefacts to disk, and optionally push to HuggingFace Hub.

        Persisted artefacts
        -------------------
        * ``config.json``              — HF-compatible adapter config: ``adapter_type``,
          ``base_model_id``, ``fitted_layers``, ``k_values``, and architecture metadata.
          Follows the same structural convention as PEFT adapter configs so that
          adapters are self-describing and discoverable on the Hub.
        * ``directions.safetensors``   — PCA component tensors and mean-diff vectors,
          keyed as ``"{layer_name}/components"`` and ``"{layer_name}/mean_diff"``.
        * ``directions_meta.json``     — non-tensor per-layer metadata (explained
          variance ratios, n_pairs_fit, k_values).

        Model weights are **not** saved — the adapter is a lightweight side-file
        (~1--5 MB) that references the base model by HuggingFace model ID.

        Parameters
        ----------
        path:
            Local directory path.  Created automatically if it does not exist.
        push_to_hub:
            If ``True``, upload the saved artefacts to HuggingFace Hub using
            ``huggingface_hub``.  Requires ``HF_TOKEN`` to be set or a prior
            ``huggingface-cli login``.
        repo_id:
            HuggingFace Hub repository ID (e.g. ``"Kameshr/sycophancy-llama"``).
            Required when ``push_to_hub=True``.
        private:
            If ``True``, create the Hub repo as private.

        Raises
        ------
        RuntimeError
            If :py:meth:`fit` has not been called.
        ValueError
            If ``push_to_hub=True`` but ``repo_id`` is not provided.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save: Baker has not been fitted.")
        if push_to_hub and not repo_id:
            raise ValueError("repo_id is required when push_to_hub=True.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._director.save(
            directions=self._directions,
            path=str(out_dir / "directions.safetensors"),
        )

        # HuggingFace-compatible adapter config.  The ``adapter_type`` field
        # mirrors the PEFT convention so the file is self-describing on the Hub.
        config: Dict[str, Any] = {
            "adapter_type": "activation_baking",
            "adapter_version": "1.0",
            "base_model_id": self._model_id,
            "fitted_layers": self._fitted_layers,
            "k_values": self._k_values,
            "n_components": next(
                iter(self._directions.values())
            ).components.shape[0] if self._directions else 0,
            "model_info": {
                "architecture": self._model_info.architecture,
                "num_layers": self._model_info.num_layers,
                "hidden_size": self._model_info.hidden_size,
            },
        }
        config_path = out_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

        logger.info("Baker artefacts saved to '%s'.", out_dir)

        if push_to_hub:
            try:
                from huggingface_hub import HfApi  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required for push_to_hub. "
                    "Install with: pip install huggingface_hub"
                ) from exc

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info("Adapter pushed to HuggingFace Hub: https://huggingface.co/%s", repo_id)

    # ------------------------------------------------------------------
    # Weight fusion
    # ------------------------------------------------------------------

    def fuse_to_model(self, alpha: float = 1.0) -> PreTrainedModel:
        """
        Bake steering vectors permanently into model weights.

        For each fitted layer the calibrated steering vector is added to the
        ``down_proj`` bias of that layer's MLP sub-block.  Because ``down_proj``
        is the final linear projection in the MLP, adding a constant to its
        output is mathematically equivalent to adding that constant to the
        whole block's residual-stream output — which is exactly what the
        forward hook does at inference time.

        The fused model is a complete, standard HuggingFace
        ``PreTrainedModel`` that generates with the behaviour *without* any
        hook machinery.  It can be saved with ``save_pretrained`` and loaded
        with ``AutoModelForCausalLM.from_pretrained`` like any other
        checkpoint.

        Parameters
        ----------
        alpha:
            Global scale factor applied on top of the per-layer calibrated K.
            Defaults to ``1.0`` (the same scale used by ``Baker.generate``).

        Returns
        -------
        PreTrainedModel
            A deep copy of the base model with fused biases.  The original
            ``Baker`` instance is not modified.

        Raises
        ------
        RuntimeError
            If :py:meth:`fit` has not been called.
        ValueError
            If a fitted layer name cannot be mapped to an ``mlp_down_proj``
            module (i.e. the layer name is not in ``model_info.layer_module_names``).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot fuse: Baker has not been fitted.")

        logger.info(
            "Fusing steering vectors into model weights (alpha=%.4f, %d layers).",
            alpha,
            len(self._directions),
        )
        model_copy: PreTrainedModel = copy.deepcopy(self._model)
        model_copy.eval()

        # Tell HF that this checkpoint has MLP biases so they survive save/load.
        # Without this, from_pretrained creates Linear layers without bias and
        # silently discards the saved bias tensors.
        if hasattr(model_copy.config, "mlp_bias"):
            model_copy.config.mlp_bias = True

        # Zero-initialise biases on every down_proj so all layers round-trip
        # cleanly; only fitted layers will receive a non-zero steering delta.
        with torch.no_grad():
            for down_proj_name in self._model_info.mlp_down_proj_names:
                down_proj: nn.Linear = get_layer_module(model_copy, down_proj_name)  # type: ignore[assignment]
                if down_proj.bias is None:
                    down_proj.bias = nn.Parameter(
                        torch.zeros(
                            down_proj.out_features,
                            device=down_proj.weight.device,
                            dtype=down_proj.weight.dtype,
                        ),
                        requires_grad=False,
                    )

        layer_to_idx: Dict[str, int] = {
            name: idx
            for idx, name in enumerate(self._model_info.layer_module_names)
        }

        fused_count = 0
        with torch.no_grad():
            for layer_name, bd in self._directions.items():
                if layer_name not in layer_to_idx:
                    logger.warning(
                        "fuse_to_model: layer '%s' not found in model_info; skipping.",
                        layer_name,
                    )
                    continue
                if bd.k_value is None:
                    logger.warning(
                        "fuse_to_model: k_value is None for layer '%s'; skipping.",
                        layer_name,
                    )
                    continue

                layer_idx = layer_to_idx[layer_name]
                down_proj_name = self._model_info.mlp_down_proj_names[layer_idx]

                # Compute the steering vector in float32 on CPU.
                # Formula: alpha * k * components^T @ (components @ mean_diff)
                components = bd.components.float().cpu()   # [k, hidden]
                mean_diff = bd.mean_diff.float().cpu()     # [hidden]
                projection_weights = torch.mv(components, mean_diff)       # [k]
                steering_vector = torch.mv(components.T, projection_weights)  # [hidden]
                bias_delta = (alpha * bd.k_value * steering_vector)        # [hidden]

                # Retrieve the down_proj Linear module from the copy.
                down_proj: nn.Linear = get_layer_module(model_copy, down_proj_name)  # type: ignore[assignment]
                target_dtype = down_proj.weight.dtype
                target_device = down_proj.weight.device
                bias_delta = bias_delta.to(device=target_device, dtype=target_dtype)

                if down_proj.bias is None:
                    # Create a new bias parameter initialised to the steering vector.
                    down_proj.bias = nn.Parameter(
                        bias_delta.clone(), requires_grad=False
                    )
                else:
                    down_proj.bias.data.add_(bias_delta)

                fused_count += 1
                logger.debug(
                    "Fused steering into '%s' (bias_delta norm=%.4e).",
                    down_proj_name,
                    bias_delta.norm().item(),
                )

        gc.collect()
        logger.info("Weight fusion complete: %d layers fused.", fused_count)
        return model_copy

    def save_fused_model(
        self,
        path: str,
        alpha: float = 1.0,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        private: bool = False,
    ) -> PreTrainedModel:
        """
        Fuse steering vectors and save the resulting model + tokenizer.

        Calls :py:meth:`fuse_to_model`, then writes the fused model and its
        tokenizer to ``path`` via ``save_pretrained``.  Optionally pushes both
        to the HuggingFace Hub so the adapter can be loaded by anyone with a
        plain ``AutoModelForCausalLM.from_pretrained`` call — no
        ``activation_baking`` library required.

        A ``fused_adapter_config.json`` is also written alongside the model
        to record provenance (base model, alpha, K values, layer count).

        Parameters
        ----------
        path:
            Local directory to write the fused model and tokenizer into.
        alpha:
            Steering magnitude multiplier forwarded to :py:meth:`fuse_to_model`.
        push_to_hub:
            Upload the saved artefacts to HuggingFace Hub.
        repo_id:
            HuggingFace Hub repository ID (e.g. ``"Kameshr/sycophancy-llama-fused"``).
            Required when ``push_to_hub=True``.
        private:
            If ``True``, create the Hub repo as private.

        Returns
        -------
        PreTrainedModel
            The fused model (already saved; returned for in-process use).

        Raises
        ------
        RuntimeError
            If :py:meth:`fit` has not been called.
        ValueError
            If ``push_to_hub=True`` but ``repo_id`` is not provided.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save fused model: Baker has not been fitted.")
        if push_to_hub and not repo_id:
            raise ValueError("repo_id is required when push_to_hub=True.")

        fused_model = self.fuse_to_model(alpha=alpha)

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving fused model to '%s'.", out_dir)
        fused_model.save_pretrained(str(out_dir))
        self._tokenizer.save_pretrained(str(out_dir))

        # Provenance record — not required for loading, but useful for reproducibility.
        provenance: Dict[str, Any] = {
            "fused_from": "activation_baking",
            "base_model_id": self._model_id,
            "alpha": alpha,
            "k_values": self._k_values,
            "fitted_layers": self._fitted_layers,
            "n_components": next(
                iter(self._directions.values())
            ).components.shape[0] if self._directions else 0,
            "model_info": {
                "architecture": self._model_info.architecture,
                "num_layers": self._model_info.num_layers,
                "hidden_size": self._model_info.hidden_size,
            },
        }
        prov_path = out_dir / "fused_adapter_config.json"
        with prov_path.open("w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)
        logger.info("Provenance config saved → %s", prov_path)

        if push_to_hub:
            try:
                from huggingface_hub import HfApi  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required for push_to_hub. "
                    "Install with: pip install huggingface_hub"
                ) from exc

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(
                "Fused model pushed to HuggingFace Hub: https://huggingface.co/%s",
                repo_id,
            )

        return fused_model

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "Baker":
        """
        Load a previously saved :class:`Baker` instance.

        ``path`` may be either a local directory written by :py:meth:`save` **or**
        a HuggingFace Hub repository ID (e.g. ``"Kameshr/sycophancy-llama"``).
        When a Hub ID is provided the artefacts are downloaded automatically via
        ``huggingface_hub``; the base model is always streamed from the Hub.

        This mirrors the PEFT ``PeftModel.from_pretrained`` pattern: the adapter
        config is self-describing (it embeds ``base_model_id``), so no additional
        arguments are needed to reconstruct the full model + adapter stack.

        Parameters
        ----------
        path:
            Local directory path **or** HuggingFace Hub repo ID.
        device:
            Target device for the reloaded model.

        Returns
        -------
        Baker
            A fitted Baker ready for generation.

        Raises
        ------
        FileNotFoundError
            If the required artefact files are missing locally.

        Examples
        --------
        Load from local disk::

            baker = Baker.load("./my_adapter")

        Load from HuggingFace Hub::

            baker = Baker.load("Kameshr/sycophancy-suppression-llama")
        """
        in_dir = Path(path)

        # Attempt Hub download when the path is not a local directory
        if not in_dir.is_dir():
            try:
                from huggingface_hub import snapshot_download  # type: ignore[import]
                logger.info("Path '%s' is not a local directory — attempting Hub download.", path)
                local_dir = snapshot_download(repo_id=path, repo_type="model")
                in_dir = Path(local_dir)
            except Exception as exc:
                raise FileNotFoundError(
                    f"'{path}' is not a local directory and Hub download failed: {exc}"
                ) from exc

        config_path = in_dir / "config.json"
        # Prefer safetensors; fall back to legacy pickle for backward compatibility.
        directions_path = in_dir / "directions.safetensors"
        if not directions_path.exists():
            directions_path = in_dir / "directions.pkl"

        for p in (config_path, directions_path):
            if not p.exists():
                raise FileNotFoundError(f"Required artefact not found: '{p}'")

        with config_path.open("r", encoding="utf-8") as fh:
            config: Dict[str, Any] = json.load(fh)

        # Support both old format (model_id key) and new HF-compatible format (base_model_id)
        model_id: str = config.get("base_model_id", config.get("model_id", ""))
        if not model_id:
            raise ValueError("config.json missing 'base_model_id' field.")

        logger.info("Loading Baker from '%s' (base_model_id='%s').", in_dir, model_id)

        baker = cls(model_id=model_id, device=device)
        baker._directions = PCADirector.load(str(directions_path))
        baker._k_values = config.get("k_values", {})
        baker._fitted_layers = config.get("fitted_layers", list(baker._directions.keys()))
        baker._is_fitted = True

        logger.info("Baker loaded from '%s'.", in_dir)
        return baker
