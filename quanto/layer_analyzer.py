"""
Layer Analyzer: Automatically detect sensitive layers that should be excluded from quantization.

This module analyzes model architecture to identify layers that are sensitive to quantization
and should be excluded to maintain model quality.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from quark.torch import LLMTemplate


@dataclass
class LayerAnalysisResult:
    """Result of layer analysis containing exclusion recommendations."""

    exclude_patterns: list[str] = field(default_factory=list)
    layer_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        result = ["Layer Analysis Results:", "=" * 50]
        result.append(f"\nRecommended exclusion patterns: {self.exclude_patterns}")
        result.append("\nReasoning:")
        for pattern, reason in self.reasoning.items():
            result.append(f"  - {pattern}: {reason}")
        return "\n".join(result)


class LayerAnalyzer:
    """
    Analyzes model architecture to detect sensitive layers.

    This class examines the model structure and identifies layers that typically
    need to be excluded from quantization for optimal model quality.
    """

    # Default patterns for layers that should always be excluded
    ALWAYS_EXCLUDE_PATTERNS = [
        "lm_head",  # Output projection - critical for token generation
    ]

    # Patterns for layers that are often sensitive (model-specific)
    SENSITIVE_PATTERNS = {
        # Gate layers in MoE models (not standard Llama gate_proj)
        "*mlp.gate",  # Exact MoE gate layers
        "*shared_expert_gate*",
        # Linear attention projections
        "*linear_attn*",
    }

    # Last layer patterns (often more sensitive)
    LAST_LAYER_PATTERNS = [
        r"model\.layers\.(\d+)\..*",
    ]

    def __init__(
        self,
        model: nn.Module,
        model_type: str | None = None,
        template: LLMTemplate | None = None,
    ):
        """
        Initialize the layer analyzer.

        Args:
            model: The PyTorch model to analyze
            model_type: Optional model type string (e.g., 'llama', 'qwen2')
            template: Optional LLMTemplate for the model
        """
        self.model = model
        self.model_type = model_type
        self.template = template
        self._layer_info: dict[str, dict[str, Any]] = {}
        self._total_layers = 0

    def _get_all_linear_layers(self) -> dict[str, nn.Linear]:
        """Get all Linear layers in the model."""
        linear_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers

    def _get_layer_dimensions(self, layer: nn.Linear) -> tuple[int, int]:
        """Get input and output dimensions of a linear layer."""
        return layer.in_features, layer.out_features

    def _is_small_dimension(self, in_features: int, out_features: int, threshold: int = 4096) -> bool:
        """Check if a layer has small dimensions that might be sensitive."""
        return min(in_features, out_features) < threshold

    def _detect_gate_layers(self, layer_names: list[str]) -> list[str]:
        """Detect MoE gate layers (not standard Llama gate_proj which is a normal MLP layer)."""
        # MoE gate patterns - these are specific to MoE architectures
        # Standard Llama gate_proj is NOT a gate layer, it's just part of the MLP
        gate_patterns = ["*mlp.gate$", "*gate$", "*router*", "*shared_expert_gate*"]
        gates = []
        for name in layer_names:
            # Skip gate_proj which is a normal Llama MLP layer
            if "gate_proj" in name:
                continue
            for pattern in gate_patterns:
                if fnmatch.fnmatch(name, pattern) or pattern.replace("*", "").replace("$", "") in name.lower():
                    gates.append(name)
                    break
        return gates

    def _detect_last_layers(self, layer_names: list[str]) -> list[str]:
        """Detect the last N transformer layers (often more sensitive)."""
        # Extract layer numbers from layer names
        layer_nums = set()
        layer_pattern = re.compile(r"model\.layers\.(\d+)")

        for name in layer_names:
            match = layer_pattern.search(name)
            if match:
                layer_nums.add(int(match.group(1)))

        if not layer_nums:
            return []

        max_layer = max(layer_nums)

        # The last layer is often sensitive
        last_layer_patterns = [f"model.layers.{max_layer}.*"]
        return last_layer_patterns

    def _get_template_exclude_layers(self) -> list[str]:
        """Get exclude layers from the template if available."""
        if self.template is not None:
            return list(self.template.exclude_layers_name)
        return []

    def _analyze_layer_weights(self, linear_layers: dict[str, nn.Linear]) -> dict[str, float]:
        """
        Analyze layer weights to detect sensitive layers.

        Layers with unusual weight distributions may be more sensitive to quantization.
        """
        sensitivity_scores = {}

        for name, layer in linear_layers.items():
            if layer.weight is None:
                continue

            weight = layer.weight.data.float()
            std = weight.std().item()
            mean = weight.mean().item()
            abs_mean = weight.abs().mean().item()

            # High variance or unusual distributions may indicate sensitivity
            # These are heuristics and may need tuning
            if std > 0.1:  # High variance
                sensitivity_scores[name] = std

        return sensitivity_scores

    def analyze(
        self,
        aggressive: bool = False,
        exclude_last_n_layers: int = 0,
        custom_patterns: list[str] | None = None,
    ) -> LayerAnalysisResult:
        """
        Analyze the model and recommend layers to exclude.

        Args:
            aggressive: If True, exclude more layers (for aggressive quality preservation)
            exclude_last_n_layers: Number of last transformer layers to exclude (0 = auto-detect)
            custom_patterns: Additional custom patterns to exclude

        Returns:
            LayerAnalysisResult with recommended exclusion patterns
        """
        exclude_patterns = []
        reasoning = {}

        # Start with always-exclude patterns
        for pattern in self.ALWAYS_EXCLUDE_PATTERNS:
            exclude_patterns.append(pattern)
            reasoning[pattern] = "Always excluded (critical for model output)"

        # Get template-based exclusions
        template_excludes = self._get_template_exclude_layers()
        for pattern in template_excludes:
            if pattern not in exclude_patterns:
                exclude_patterns.append(pattern)
                reasoning[pattern] = "Recommended by model template"

        # Get all linear layers
        linear_layers = self._get_all_linear_layers()
        layer_names = list(linear_layers.keys())

        # Detect gate layers (MoE models)
        gate_layers = self._detect_gate_layers(layer_names)
        for gate in gate_layers:
            pattern = f"*{gate}*"
            if pattern not in exclude_patterns:
                exclude_patterns.append(gate)
                reasoning[gate] = "MoE gate/router layer (sensitive to quantization)"

        # Detect attention layers with small dimensions
        if aggressive:
            for name, layer in linear_layers.items():
                in_f, out_f = self._get_layer_dimensions(layer)
                if self._is_small_dimension(in_f, out_f, threshold=2048):
                    if "attn" in name.lower() or "attention" in name.lower():
                        pattern = f"*{name}*"
                        if pattern not in exclude_patterns:
                            exclude_patterns.append(pattern)
                            reasoning[pattern] = f"Small dimension attention layer ({in_f}x{out_f})"

        # Handle last layers
        if exclude_last_n_layers > 0:
            # Get layer numbers
            layer_nums = set()
            layer_pattern = re.compile(r"model\.layers\.(\d+)")
            for name in layer_names:
                match = layer_pattern.search(name)
                if match:
                    layer_nums.add(int(match.group(1)))

            if layer_nums:
                max_layer = max(layer_nums)
                for i in range(max(0, max_layer - exclude_last_n_layers + 1), max_layer + 1):
                    pattern = f"model.layers.{i}.*"
                    if pattern not in exclude_patterns:
                        exclude_patterns.append(pattern)
                        reasoning[pattern] = f"Last {exclude_last_n_layers} transformer layers (often sensitive)"
        else:
            # Auto-detect if we should exclude the very last layer
            last_layer_patterns = self._detect_last_layers(layer_names)
            for pattern in last_layer_patterns:
                if pattern not in exclude_patterns:
                    exclude_patterns.append(pattern)
                    reasoning[pattern] = "Last transformer layer (often sensitive)"

        # Add custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                if pattern not in exclude_patterns:
                    exclude_patterns.append(pattern)
                    reasoning[pattern] = "Custom exclusion pattern"

        # Build layer info
        layer_info = {}
        for name, layer in linear_layers.items():
            in_f, out_f = self._get_layer_dimensions(layer)
            layer_info[name] = {
                "in_features": in_f,
                "out_features": out_f,
                "is_excluded": any(fnmatch.fnmatch(name, p) for p in exclude_patterns),
            }

        return LayerAnalysisResult(
            exclude_patterns=exclude_patterns,
            layer_info=layer_info,
            reasoning=reasoning,
        )

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        linear_layers = self._get_all_linear_layers()

        summary = ["Model Architecture Summary", "=" * 50]
        summary.append(f"Total Linear Layers: {len(linear_layers)}")

        # Group by layer type
        layer_types: dict[str, int] = {}
        for name in linear_layers:
            # Extract layer type from name
            parts = name.split(".")
            if len(parts) >= 2:
                layer_type = parts[-1] if "_" in parts[-1] else parts[-2]
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

        summary.append("\nLayer Types:")
        for lt, count in sorted(layer_types.items()):
            summary.append(f"  {lt}: {count}")

        # Dimension statistics
        dims = [self._get_layer_dimensions(layer) for layer in linear_layers.values()]
        if dims:
            in_dims = [d[0] for d in dims]
            out_dims = [d[1] for d in dims]
            summary.append(f"\nDimension Statistics:")
            summary.append(f"  Input features: min={min(in_dims)}, max={max(in_dims)}, avg={sum(in_dims)//len(in_dims)}")
            summary.append(f"  Output features: min={min(out_dims)}, max={max(out_dims)}, avg={sum(out_dims)//len(out_dims)}")

        return "\n".join(summary)
