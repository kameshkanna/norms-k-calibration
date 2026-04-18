"""
activation_baking/model_utils.py

Utilities for detecting model architecture, navigating submodules, and applying
neuron permutations for weight-space symmetry experiments in activation steering
research.
"""

import copy
import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture registry: maps architecture tag -> (layer_path, mlp_down, attn_o)
# Each entry is a callable(num_layers) -> List[str]
# ---------------------------------------------------------------------------
_ARCH_PATTERNS: Dict[str, Dict[str, str]] = {
    "llama": {
        "layer_prefix": "model.layers",
        "mlp_down_proj": "mlp.down_proj",
        "mlp_up_proj": "mlp.up_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "attn_o_proj": "self_attn.o_proj",
        "attn_q_proj": "self_attn.q_proj",
        "attn_k_proj": "self_attn.k_proj",
        "attn_v_proj": "self_attn.v_proj",
    },
    "qwen2": {
        "layer_prefix": "model.layers",
        "mlp_down_proj": "mlp.down_proj",
        "mlp_up_proj": "mlp.up_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "attn_o_proj": "self_attn.o_proj",
        "attn_q_proj": "self_attn.q_proj",
        "attn_k_proj": "self_attn.k_proj",
        "attn_v_proj": "self_attn.v_proj",
    },
    "gemma2": {
        "layer_prefix": "model.layers",
        "mlp_down_proj": "mlp.down_proj",
        "mlp_up_proj": "mlp.up_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "attn_o_proj": "self_attn.o_proj",
        "attn_q_proj": "self_attn.q_proj",
        "attn_k_proj": "self_attn.k_proj",
        "attn_v_proj": "self_attn.v_proj",
    },
    "mistral": {
        "layer_prefix": "model.layers",
        "mlp_down_proj": "mlp.down_proj",
        "mlp_up_proj": "mlp.up_proj",
        "mlp_gate_proj": "mlp.gate_proj",
        "attn_o_proj": "self_attn.o_proj",
        "attn_q_proj": "self_attn.q_proj",
        "attn_k_proj": "self_attn.k_proj",
        "attn_v_proj": "self_attn.v_proj",
    },
}


@dataclass
class ModelInfo:
    """Container for structural metadata about a transformer model.

    Attributes:
        model_id: HuggingFace model identifier string.
        architecture: Canonical architecture family name (llama, qwen2, gemma2, mistral).
        num_layers: Total number of transformer decoder layers.
        hidden_size: Hidden / residual stream dimensionality.
        is_instruct: True when the checkpoint is instruction-tuned.
        layer_module_names: Dot-separated paths to each decoder block module,
            e.g. ["model.layers.0", "model.layers.1", ...].
        mlp_down_proj_names: Dot-separated paths to down_proj Linear modules,
            one per layer.
        attn_out_proj_names: Dot-separated paths to o_proj Linear modules,
            one per layer.
        arch_patterns: Internal mapping of sub-module path fragments for the
            detected architecture, used by permutation utilities.
    """

    model_id: str
    architecture: str
    num_layers: int
    hidden_size: int
    is_instruct: bool
    layer_module_names: List[str]
    mlp_down_proj_names: List[str]
    attn_out_proj_names: List[str]
    arch_patterns: Dict[str, str] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_model_info(model: PreTrainedModel, model_id: str) -> ModelInfo:
    """Inspect a loaded model and return a fully-populated ModelInfo.

    Detection strategy:
    1. Check model.config.model_type against known architecture names.
    2. Fall back to inspecting named modules for canonical sub-module patterns.

    Args:
        model: A loaded HuggingFace PreTrainedModel.
        model_id: The HuggingFace model identifier (used for is_instruct detection
            and stored in ModelInfo.model_id).

    Returns:
        ModelInfo populated with layer counts, hidden size, and all module paths.

    Raises:
        ValueError: If the architecture cannot be resolved to a supported family.
        AttributeError: If expected config attributes are absent.
    """
    if not isinstance(model, PreTrainedModel):
        raise TypeError(
            f"Expected a PreTrainedModel instance, got {type(model).__name__}."
        )
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string.")

    architecture = _resolve_architecture(model)
    patterns = _ARCH_PATTERNS[architecture]

    config = model.config
    num_layers: int = _get_num_layers(config, architecture)
    hidden_size: int = _get_hidden_size(config, architecture)

    layer_prefix = patterns["layer_prefix"]
    layer_module_names = [f"{layer_prefix}.{i}" for i in range(num_layers)]
    mlp_down_proj_names = [
        f"{layer_prefix}.{i}.{patterns['mlp_down_proj']}" for i in range(num_layers)
    ]
    attn_out_proj_names = [
        f"{layer_prefix}.{i}.{patterns['attn_o_proj']}" for i in range(num_layers)
    ]

    is_instruct = _detect_instruct(model_id)

    info = ModelInfo(
        model_id=model_id,
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        is_instruct=is_instruct,
        layer_module_names=layer_module_names,
        mlp_down_proj_names=mlp_down_proj_names,
        attn_out_proj_names=attn_out_proj_names,
        arch_patterns=patterns,
    )
    logger.info(
        "Detected model info: arch=%s, layers=%d, hidden=%d, instruct=%s",
        architecture,
        num_layers,
        hidden_size,
        is_instruct,
    )
    return info


def get_layer_module(model: PreTrainedModel, module_name: str) -> nn.Module:
    """Traverse a model's module hierarchy using a dot-separated path.

    Args:
        model: The root PreTrainedModel to navigate.
        module_name: Dot-separated attribute path, e.g. "model.layers.3.mlp.down_proj".

    Returns:
        The nn.Module at the specified path.

    Raises:
        ValueError: If module_name is empty or malformed.
        AttributeError: If any segment of the path does not exist on the parent.
    """
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty dot-separated string.")

    segments = module_name.split(".")
    current: nn.Module = model
    for seg in segments:
        if seg.isdigit():
            try:
                current = current[int(seg)]  # type: ignore[index]
            except (IndexError, TypeError) as exc:
                raise AttributeError(
                    f"Cannot index module with '{seg}' on {type(current).__name__}."
                ) from exc
        else:
            if not hasattr(current, seg):
                raise AttributeError(
                    f"Module '{type(current).__name__}' has no attribute '{seg}' "
                    f"(full path: '{module_name}')."
                )
            current = getattr(current, seg)
    return current


def apply_neuron_permutation(
    model: PreTrainedModel,
    model_info: ModelInfo,
    layer_indices: List[int],
    seed: int,
) -> PreTrainedModel:
    """Return a deep copy of the model with consistent neuron permutations applied.

    For each specified layer the function permutes the neuron ordering in both the
    MLP (up/gate/down projections) and attention (q/k/v/o projections) sub-networks
    such that the model computes an identical function — only the internal neuron
    indices are reordered.  This is used to verify that activation PCA directions
    derived from contrastive prompts are invariant to weight-space symmetries.

    MLP permutation (SwiGLU / Gated-linear):
        Let P be a permutation of [0, intermediate_size).
        - up_proj:   W_up[P, :]         (permute output rows)
        - gate_proj: W_gate[P, :]       (permute output rows, same P)
        - down_proj: W_down[:, P]       (permute input columns to match)

    Attention permutation:
        Let P_head be a permutation of [0, head_dim * n_heads).
        - q_proj:  W_q[P_head, :]
        - k_proj:  W_k[P_kv, :]         (separate permutation for KV if n_kv_heads differs)
        - v_proj:  W_v[P_kv, :]
        - o_proj:  W_o[:, P_head]

    Args:
        model: Source PreTrainedModel (not modified in place).
        model_info: ModelInfo describing the model's architecture.
        layer_indices: List of zero-based layer indices to permute.
        seed: Random seed for reproducible permutations.

    Returns:
        A deep copy of the model with permuted weights at the specified layers.

    Raises:
        ValueError: If any layer index is out of range.
        TypeError: If layer_indices is not a list of integers.
    """
    if not isinstance(layer_indices, list) or not all(
        isinstance(i, int) for i in layer_indices
    ):
        raise TypeError("layer_indices must be a list of integers.")

    invalid = [i for i in layer_indices if not (0 <= i < model_info.num_layers)]
    if invalid:
        raise ValueError(
            f"Layer indices {invalid} are out of range [0, {model_info.num_layers})."
        )

    logger.info(
        "Creating deep copy of model for permutation experiment (seed=%d, layers=%s).",
        seed,
        layer_indices,
    )
    model_copy: PreTrainedModel = copy.deepcopy(model)
    patterns = model_info.arch_patterns
    layer_prefix = patterns["layer_prefix"]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for layer_idx in layer_indices:
        base = f"{layer_prefix}.{layer_idx}"
        _permute_mlp_layer(model_copy, base, patterns, rng)
        _permute_attn_layer(model_copy, base, patterns, rng, model_copy.config)

    gc.collect()
    logger.info("Permutation complete for %d layers.", len(layer_indices))
    return model_copy


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_architecture(model: PreTrainedModel) -> str:
    """Determine the canonical architecture family for a model.

    Args:
        model: A loaded HuggingFace PreTrainedModel.

    Returns:
        One of {"llama", "qwen2", "gemma2", "mistral"}.

    Raises:
        ValueError: If the architecture cannot be mapped to a supported family.
    """
    model_type: str = getattr(model.config, "model_type", "").lower()

    # Direct match
    for arch_key in _ARCH_PATTERNS:
        if arch_key in model_type:
            return arch_key

    # Broader substring matching for variant names
    _MODEL_TYPE_MAP = {
        "llama": "llama",
        "llama2": "llama",
        "llama3": "llama",
        "codellama": "llama",
        "qwen": "qwen2",
        "qwen2": "qwen2",
        "gemma": "gemma2",
        "gemma2": "gemma2",
        "mistral": "mistral",
        "mixtral": "mistral",
    }
    for key, arch in _MODEL_TYPE_MAP.items():
        if key in model_type:
            return arch

    # Last resort: inspect module names
    named_modules = {name for name, _ in model.named_modules()}
    if any("self_attn.o_proj" in n for n in named_modules):
        if any("mlp.gate_proj" in n for n in named_modules):
            logger.warning(
                "Architecture '%s' not in registry; defaulting to 'llama' pattern.",
                model_type,
            )
            return "llama"

    raise ValueError(
        f"Unsupported model architecture: '{model_type}'. "
        f"Supported families: {list(_ARCH_PATTERNS.keys())}."
    )


def _get_num_layers(config, architecture: str) -> int:
    """Extract the number of transformer layers from a model config.

    Args:
        config: HuggingFace model configuration object.
        architecture: Canonical architecture family string.

    Returns:
        Integer number of decoder layers.

    Raises:
        AttributeError: If no recognised attribute for layer count exists.
    """
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    raise AttributeError(
        f"Cannot determine num_layers from config for architecture '{architecture}'. "
        f"Tried: num_hidden_layers, n_layer, num_layers, n_layers."
    )


def _get_hidden_size(config, architecture: str) -> int:
    """Extract the hidden / residual stream dimension from a model config.

    Args:
        config: HuggingFace model configuration object.
        architecture: Canonical architecture family string.

    Returns:
        Integer hidden size.

    Raises:
        AttributeError: If no recognised attribute for hidden size exists.
    """
    for attr in ("hidden_size", "d_model", "n_embd"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    raise AttributeError(
        f"Cannot determine hidden_size from config for architecture '{architecture}'. "
        f"Tried: hidden_size, d_model, n_embd."
    )


def _detect_instruct(model_id: str) -> bool:
    """Return True if the model_id indicates an instruction-tuned checkpoint.

    Args:
        model_id: HuggingFace model identifier string.

    Returns:
        Boolean flag.
    """
    lowered = model_id.lower()
    return any(tag in lowered for tag in ("instruct", "-it", "-chat", "chat-"))


def _permute_mlp_layer(
    model: PreTrainedModel,
    base_path: str,
    patterns: Dict[str, str],
    rng: torch.Generator,
) -> None:
    """Apply a consistent intermediate-dimension permutation to one MLP block in-place.

    Args:
        model: The model copy being modified.
        base_path: Dot-separated path prefix for the current layer.
        patterns: Architecture-specific sub-module path fragments.
        rng: Seeded torch Generator for reproducibility.
    """
    up_path = f"{base_path}.{patterns['mlp_up_proj']}"
    gate_path = f"{base_path}.{patterns['mlp_gate_proj']}"
    down_path = f"{base_path}.{patterns['mlp_down_proj']}"

    up_module: nn.Linear = get_layer_module(model, up_path)  # type: ignore[assignment]
    gate_module: nn.Linear = get_layer_module(model, gate_path)  # type: ignore[assignment]
    down_module: nn.Linear = get_layer_module(model, down_path)  # type: ignore[assignment]

    intermediate_size = up_module.weight.shape[0]
    perm = torch.randperm(intermediate_size, generator=rng, device="cpu")

    with torch.no_grad():
        up_module.weight.data = up_module.weight.data[perm, :]
        if up_module.bias is not None:
            up_module.bias.data = up_module.bias.data[perm]

        gate_module.weight.data = gate_module.weight.data[perm, :]
        if gate_module.bias is not None:
            gate_module.bias.data = gate_module.bias.data[perm]

        down_module.weight.data = down_module.weight.data[:, perm]
        # down_proj bias is not permuted (it's in the output space)


def _permute_attn_layer(
    model: PreTrainedModel,
    base_path: str,
    patterns: Dict[str, str],
    rng: torch.Generator,
    config,
) -> None:
    """Apply a consistent head-dimension permutation to one attention block in-place.

    For models with grouped-query attention (n_kv_heads != n_heads) a separate
    permutation is applied to KV projections in their own output space.

    Args:
        model: The model copy being modified.
        base_path: Dot-separated path prefix for the current layer.
        patterns: Architecture-specific sub-module path fragments.
        rng: Seeded torch Generator for reproducibility.
        config: Model configuration (used to read head counts).
    """
    q_path = f"{base_path}.{patterns['attn_q_proj']}"
    k_path = f"{base_path}.{patterns['attn_k_proj']}"
    v_path = f"{base_path}.{patterns['attn_v_proj']}"
    o_path = f"{base_path}.{patterns['attn_o_proj']}"

    q_module: nn.Linear = get_layer_module(model, q_path)  # type: ignore[assignment]
    k_module: nn.Linear = get_layer_module(model, k_path)  # type: ignore[assignment]
    v_module: nn.Linear = get_layer_module(model, v_path)  # type: ignore[assignment]
    o_module: nn.Linear = get_layer_module(model, o_path)  # type: ignore[assignment]

    q_out_dim = q_module.weight.shape[0]
    kv_out_dim = k_module.weight.shape[0]

    perm_q = torch.randperm(q_out_dim, generator=rng, device="cpu")
    # Use a separate permutation for KV projections (may differ in GQA models)
    perm_kv = torch.randperm(kv_out_dim, generator=rng, device="cpu")

    with torch.no_grad():
        q_module.weight.data = q_module.weight.data[perm_q, :]
        if q_module.bias is not None:
            q_module.bias.data = q_module.bias.data[perm_q]

        k_module.weight.data = k_module.weight.data[perm_kv, :]
        if k_module.bias is not None:
            k_module.bias.data = k_module.bias.data[perm_kv]

        v_module.weight.data = v_module.weight.data[perm_kv, :]
        if v_module.bias is not None:
            v_module.bias.data = v_module.bias.data[perm_kv]

        # o_proj input space must match q permutation
        o_module.weight.data = o_module.weight.data[:, perm_q]
