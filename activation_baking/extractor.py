"""
activation_baking/extractor.py

Hook-based activation extractor for transformer residual stream outputs.
Supports batched extraction, contrastive difference computation, and per-layer
norm measurement used in K-calibration.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activation_baking.model_utils import ModelInfo, get_layer_module

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device, supporting 'auto'.

    'auto' selects CUDA if available, then MPS, then falls back to CPU.

    Args:
        device_str: One of 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'.

    Returns:
        A torch.device instance.

    Raises:
        ValueError: If the string is not a recognised device specifier.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device string '{device_str}': {exc}") from exc


class ActivationExtractor:
    """Hook-based activation extractor for transformer residual stream outputs.

    Registers forward hooks on specified decoder block modules to capture
    hidden states after the residual add at the end of each block.  Hooks are
    attached only during a forward pass and removed immediately afterwards to
    avoid memory leaks.

    Attributes:
        model: The transformer model being probed.
        tokenizer: Tokenizer paired with the model.
        model_info: Structural metadata describing the model.
        device: Resolved torch.device for inference.
        batch_size: Number of prompts processed per forward pass.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_info: ModelInfo,
        device: str = "auto",
        batch_size: int = 4,
    ) -> None:
        """Initialise the extractor.

        Args:
            model: A loaded, eval-mode PreTrainedModel.
            tokenizer: Tokenizer whose pad/eos tokens are configured.
            model_info: ModelInfo produced by detect_model_info().
            device: Device string; 'auto' resolves to cuda > mps > cpu.
            batch_size: Number of prompts to process in each forward pass.

        Raises:
            TypeError: If model or tokenizer are wrong types.
            ValueError: If batch_size < 1 or device is invalid.
        """
        if not isinstance(model, PreTrainedModel):
            raise TypeError(
                f"model must be a PreTrainedModel, got {type(model).__name__}."
            )
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(
                f"tokenizer must be a PreTrainedTokenizerBase, "
                f"got {type(tokenizer).__name__}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self.model = model
        self.tokenizer = tokenizer
        self.model_info = model_info
        self.device: torch.device = _resolve_device(device)
        self.batch_size = batch_size

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.warning(
                "Tokenizer has no pad_token; using eos_token_id=%d as pad.",
                self.tokenizer.eos_token_id,
            )

        self.model.eval()
        self.model.to(self.device)
        logger.info(
            "ActivationExtractor initialised: device=%s, batch_size=%d.",
            self.device,
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract(
        self,
        prompts: List[str],
        layer_names: List[str],
        position: str = "last",
    ) -> Dict[str, torch.Tensor]:
        """Extract residual stream activations for a list of prompts.

        Registers temporary forward hooks on each named module, runs tokenised
        prompts through the model in batches, then collates activations.

        Args:
            prompts: Flat list of prompt strings to process.
            layer_names: List of dot-separated module paths to hook into.
                Each path should point to a decoder block (e.g. "model.layers.5").
            position: Token position strategy for aggregation:
                - "last": Hidden state of the last non-padding token.
                - "mean": Mean over all non-padding token positions.

        Returns:
            Dictionary mapping each layer name to a float32 CPU tensor of
            shape [len(prompts), hidden_size].

        Raises:
            ValueError: If prompts is empty, layer_names is empty, or position
                is not one of {"last", "mean"}.
            KeyError: If a layer_name does not correspond to a module in the model.
        """
        if not prompts:
            raise ValueError("prompts list must not be empty.")
        if not layer_names:
            raise ValueError("layer_names list must not be empty.")
        if position not in {"last", "mean"}:
            raise ValueError(
                f"position must be 'last' or 'mean', got '{position}'."
            )

        self._validate_layer_names(layer_names)

        all_activations: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}

        batches = self._make_batches(prompts)
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        for batch in tqdm(batches, total=total_batches, desc="Extracting activations", dynamic_ncols=True):
            batch_acts = self._extract_batch(batch, layer_names, position)
            for ln in layer_names:
                all_activations[ln].append(batch_acts[ln])

        result: Dict[str, torch.Tensor] = {
            ln: torch.cat(tensors, dim=0).float().cpu()
            for ln, tensors in all_activations.items()
        }

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def extract_contrastive_diffs(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-layer activation differences for matched prompt pairs.

        For each pair (p_i, n_i) computes act(p_i) - act(n_i) at the last
        non-padding token position, yielding a contrastive direction vector.

        Args:
            positive_prompts: Prompts eliciting the target behaviour.
            negative_prompts: Matched prompts eliciting the baseline behaviour.
            layer_names: Module paths to extract from (same semantics as extract()).

        Returns:
            Dictionary mapping each layer name to a float32 CPU tensor of
            shape [n_pairs, hidden_size] containing the pairwise differences.

        Raises:
            ValueError: If positive_prompts and negative_prompts differ in length
                or are empty.
        """
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError(
                f"positive_prompts ({len(positive_prompts)}) and "
                f"negative_prompts ({len(negative_prompts)}) must have the same length."
            )
        if not positive_prompts:
            raise ValueError("Prompt lists must not be empty.")

        logger.info(
            "Extracting contrastive diffs for %d pairs across %d layers.",
            len(positive_prompts),
            len(layer_names),
        )

        pos_acts = self.extract(positive_prompts, layer_names, position="last")
        neg_acts = self.extract(negative_prompts, layer_names, position="last")

        diffs: Dict[str, torch.Tensor] = {
            ln: pos_acts[ln] - neg_acts[ln] for ln in layer_names
        }

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return diffs

    def compute_layer_norms(
        self,
        prompts: List[str],
        layer_names: List[str],
    ) -> Dict[str, float]:
        """Compute the mean L2 norm of activations per layer.

        Used to derive K calibration values: K = mean_norm / sqrt(hidden_size).

        Args:
            prompts: List of prompt strings to average over.
            layer_names: Module paths to measure.

        Returns:
            Dictionary mapping each layer name to the scalar mean L2 norm
            (averaged over all prompts).

        Raises:
            ValueError: If prompts or layer_names are empty.
        """
        if not prompts:
            raise ValueError("prompts list must not be empty.")
        if not layer_names:
            raise ValueError("layer_names list must not be empty.")

        logger.info(
            "Computing layer norms for %d prompts across %d layers.",
            len(prompts),
            len(layer_names),
        )

        activations = self.extract(prompts, layer_names, position="last")

        mean_norms: Dict[str, float] = {
            ln: torch.linalg.vector_norm(acts, dim=-1).mean().item()
            for ln, acts in activations.items()
        }

        return mean_norms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_layer_names(self, layer_names: List[str]) -> None:
        """Assert each layer name maps to an existing module.

        Args:
            layer_names: Module path strings to validate.

        Raises:
            KeyError: If a name does not exist in the model.
        """
        model_module_names = {name for name, _ in self.model.named_modules()}
        for ln in layer_names:
            if ln not in model_module_names:
                raise KeyError(
                    f"Layer '{ln}' not found in model. "
                    f"Check model_info.layer_module_names."
                )

    def _make_batches(self, prompts: List[str]) -> Generator[List[str], None, None]:
        """Yield successive non-overlapping slices of prompts.

        Args:
            prompts: Full list of prompt strings.

        Yields:
            Sublists of length <= self.batch_size.
        """
        for start in range(0, len(prompts), self.batch_size):
            yield prompts[start : start + self.batch_size]

    def _extract_batch(
        self,
        batch: List[str],
        layer_names: List[str],
        position: str,
    ) -> Dict[str, torch.Tensor]:
        """Run one batch through the model with hooks and return activations.

        Args:
            batch: A list of prompt strings (length <= batch_size).
            layer_names: Module paths to hook.
            position: "last" or "mean" aggregation strategy.

        Returns:
            Dictionary mapping layer name to tensor of shape
            [len(batch), hidden_size] on CPU.
        """
        encoding = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids: torch.Tensor = encoding["input_ids"].to(self.device)
        attention_mask: torch.Tensor = encoding["attention_mask"].to(self.device)

        # Storage for hook outputs: layer_name -> [batch, seq, hidden]
        captured: Dict[str, torch.Tensor] = {}

        hooks = []
        for ln in layer_names:
            module = get_layer_module(self.model, ln)
            hook = module.register_forward_hook(
                self._make_hook(ln, captured)
            )
            hooks.append(hook)

        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for hook in hooks:
                hook.remove()

        batch_results: Dict[str, torch.Tensor] = {}
        for ln in layer_names:
            raw = captured[ln]  # [batch, seq, hidden]
            batch_results[ln] = self._aggregate_position(
                raw, attention_mask, position
            ).cpu()

        return batch_results

    @staticmethod
    def _make_hook(
        layer_name: str,
        storage: Dict[str, torch.Tensor],
    ):
        """Factory for a forward hook that captures the module output tensor.

        The hook captures the first tensor element from the module output,
        whether the output is a plain tensor or a tuple (as in HuggingFace
        decoder blocks which return (hidden_state, ...) tuples).

        Args:
            layer_name: Key under which to store the captured tensor.
            storage: Shared dictionary to write into.

        Returns:
            A callable compatible with nn.Module.register_forward_hook.
        """
        def hook(
            module: nn.Module,
            inputs: Tuple,
            output,
        ) -> None:
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            # Detach to prevent gradients accumulating in the extraction buffer
            storage[layer_name] = tensor.detach()

        return hook

    @staticmethod
    def _aggregate_position(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position: str,
    ) -> torch.Tensor:
        """Aggregate a [batch, seq, hidden] tensor along the sequence dimension.

        Args:
            hidden_states: Float tensor of shape [batch, seq, hidden].
            attention_mask: Integer tensor of shape [batch, seq] with 1/0 entries.
            position: "last" selects the last unmasked position; "mean" averages
                over all unmasked positions.

        Returns:
            Float tensor of shape [batch, hidden].
        """
        if position == "last":
            # Find index of last non-padding token per sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
            batch_size = hidden_states.size(0)
            hidden_size = hidden_states.size(2)
            idx = seq_lengths.view(-1, 1, 1).expand(batch_size, 1, hidden_size)
            return hidden_states.gather(dim=1, index=idx).squeeze(1)

        # position == "mean"
        mask_float = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        summed = (hidden_states * mask_float).sum(dim=1)   # [batch, hidden]
        counts = mask_float.sum(dim=1).clamp(min=1.0)      # [batch, 1]
        return summed / counts
