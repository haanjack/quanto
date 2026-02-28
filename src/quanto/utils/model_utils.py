"""
Quanto: General Purpose LLM Quantization Tool
Model detection and template utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig

from ..constants import MODEL_TYPE_MAPPINGS

if TYPE_CHECKING:
    from quark.torch import LLMTemplate


def detect_model_type(model_path: str, trust_remote_code: bool = True) -> str:
    """
    Detect model type from model path.

    Args:
        model_path: Path to the model directory
        trust_remote_code: Whether to trust remote code

    Returns:
        Model type string (e.g., "llama", "qwen2", "mistral")
    """
    config_path = Path(model_path) / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", config.get("architectures", ["unknown"])[0])
    else:
        # Load config from model using transformers
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", getattr(config, "architectures", ["unknown"])[0])

    return model_type


def get_template(model_type: str) -> LLMTemplate | None:
    """
    Get LLMTemplate for model type.

    Args:
        model_type: Model type string

    Returns:
        LLMTemplate instance or None if not found
    """
    # Lazy import to avoid issues if quark is not available
    from quark.torch import LLMTemplate

    available_templates = LLMTemplate.list_available()

    # Try exact match first
    if model_type in available_templates:
        return LLMTemplate.get(model_type)

    # Try common mappings
    for key, template_name in MODEL_TYPE_MAPPINGS.items():
        if key in model_type.lower():
            if template_name in available_templates:
                return LLMTemplate.get(template_name)

    # Try partial match
    for template_name in available_templates:
        if (
            model_type.lower() in template_name.lower()
            or template_name.lower() in model_type.lower()
        ):
            return LLMTemplate.get(template_name)

    return None


def get_layer_prefix(model_type: str) -> str:
    """
    Get the layer prefix for a model type.

    Args:
        model_type: Model type string

    Returns:
        Layer prefix (e.g., "model.layers")
    """
    prefixes = {
        "llama": "model.layers",
        "llama2": "model.layers",
        "llama3": "model.layers",
        "qwen": "model.layers",
        "qwen2": "model.layers",
        "qwen3": "model.layers",
        "mistral": "model.layers",
        "mixtral": "model.layers",
        "deepseek": "model.layers",
        "gemma": "model.layers",
        "gemma2": "model.layers",
        "phi": "model.layers",
        "phi3": "model.layers",
    }

    model_type_lower = model_type.lower()
    for key, prefix in prefixes.items():
        if key in model_type_lower:
            return prefix

    # Default prefix
    return "model.layers"


def get_layer_info(model_path: str, trust_remote_code: bool = True) -> dict[str, Any]:
    """
    Get layer information for a model.

    Args:
        model_path: Path to the model directory
        trust_remote_code: Whether to trust remote code

    Returns:
        Dictionary with layer info:
        - num_hidden_layers: Number of transformer layers
        - hidden_size: Hidden dimension size
        - layer_prefix: Prefix for layer names
        - model_type: Model type string
    """
    config_path = Path(model_path) / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        config = config.to_dict()

    model_type = config.get("model_type", "unknown")
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
    hidden_size = config.get("hidden_size", config.get("n_embd", 0))

    return {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "layer_prefix": get_layer_prefix(model_type),
        "model_type": model_type,
    }


def get_num_layers(model_path: str, trust_remote_code: bool = True) -> int:
    """
    Get number of transformer layers in a model.

    Args:
        model_path: Path to the model directory
        trust_remote_code: Whether to trust remote code

    Returns:
        Number of transformer layers
    """
    info = get_layer_info(model_path, trust_remote_code)
    return info["num_hidden_layers"]
