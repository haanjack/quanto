"""
Quanto: General Purpose LLM Quantization Tool
Constants and shared mappings.
"""

from __future__ import annotations

# Mapping from precision names to Quark quantization schemes
PRECISION_TO_SCHEME: dict[str, str] = {
    "int8": "int8",  # INT8 weight + activation
    "int4": "int4_wo_128",  # INT4 weight-only, group size 128 (recommended)
    "int4_64": "int4_wo_64",  # INT4 weight-only, group size 64
    "int4_32": "int4_wo_32",  # INT4 weight-only, group size 32
    "fp8": "fp8",  # FP8 weight-only
    "mxfp4": "mxfp4",  # MXFP4
    "mxfp6": "mxfp6_e3m2",  # MXFP6
    "uint4": "uint4_wo_128",  # UINT4 weight-only
}

# Default group sizes per precision
DEFAULT_GROUP_SIZES: dict[str, int] = {
    "int8": 0,  # Per-tensor
    "int4": 128,
    "int4_64": 64,
    "int4_32": 32,
    "fp8": 0,
    "mxfp4": 32,
    "mxfp6": 32,
    "uint4": 128,
}

# Model type to template mappings for LLMTemplate
MODEL_TYPE_MAPPINGS: dict[str, str] = {
    "llama": "llama",
    "llama3": "llama",
    "llama2": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen": "qwen",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "deepseek": "deepseek",
    "gemma": "gemma2",
    "gemma2": "gemma2",
    "gemma3": "gemma2",
    "phi": "phi",
    "phi3": "phi3",
    "phi4": "phi3",
}

# Default layers to exclude from quantization
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "lm_head",
]

# MoE gate layers to exclude
MOE_GATE_PATTERNS: list[str] = [
    "*.gate",  # MoE router gates (not gate_proj)
]

# Embedding and norm layers (excluded in aggressive mode)
SENSITIVE_PATTERNS: list[str] = [
    "*embed*",
    "*norm*",
]

# Supported precisions for CLI
SUPPORTED_PRECISIONS: list[str] = [
    "int8",
    "int4",
    "int4_64",
    "int4_32",
    "fp8",
    "mxfp4",
    "mxfp6",
    "uint4",
]

# Supported quantization algorithms
SUPPORTED_ALGORITHMS: list[str] = [
    "awq",
    "gptq",
    "smoothquant",
]
