"""
Layer Sensitivity Analyzer: Automatically identify layers sensitive to quantization.

This module collects layer-wise diagnostics from Quark's debug output and applies
rules to determine which layers should be excluded from quantization.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

# Add quark to path
sys.path.insert(0, str(Path(__file__).parent.parent / "quark"))

from quark.torch import LLMTemplate
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase


@dataclass
class LayerMetrics:
    """Metrics for a single layer."""

    name: str
    layer_type: str  # 'weight', 'input', 'output'

    # Weight quantization metrics
    l1_error: float | None = None  # Max absolute quantization error for weights
    shape: tuple | None = None

    # Activation quantization metrics (averaged over calibration samples)
    l1_ref_input: float | None = None  # Relative error vs reference input
    l1_ref_output: float | None = None  # Relative error vs reference output
    l1_io_error: float | None = None  # Input/output quantization error

    # Scale statistics
    scale_min: float | None = None
    scale_max: float | None = None
    has_zero_scale: bool = False

    # Computed sensitivity score
    sensitivity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "l1_error": self.l1_error,
            "shape": list(self.shape) if self.shape else None,
            "l1_ref_input": self.l1_ref_input,
            "l1_ref_output": self.l1_ref_output,
            "l1_io_error": self.l1_io_error,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "has_zero_scale": self.has_zero_scale,
            "sensitivity_score": self.sensitivity_score,
        }


@dataclass
class SensitivityAnalysisResult:
    """Result of sensitivity analysis."""

    layer_metrics: dict[str, LayerMetrics] = field(default_factory=dict)
    excluded_layers: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_metrics": {k: v.to_dict() for k, v in self.layer_metrics.items()},
            "excluded_layers": self.excluded_layers,
            "exclusion_reasons": self.exclusion_reasons,
            "statistics": self.statistics,
        }

    def save(self, path: str | Path) -> None:
        """Save analysis result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SensitivityAnalysisResult:
        """Load analysis result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        result = cls()
        for name, metrics_dict in data.get("layer_metrics", {}).items():
            metrics = LayerMetrics(
                name=metrics_dict["name"],
                layer_type=metrics_dict["layer_type"],
                l1_error=metrics_dict.get("l1_error"),
                shape=tuple(metrics_dict["shape"]) if metrics_dict.get("shape") else None,
                l1_ref_input=metrics_dict.get("l1_ref_input"),
                l1_ref_output=metrics_dict.get("l1_ref_output"),
                l1_io_error=metrics_dict.get("l1_io_error"),
                scale_min=metrics_dict.get("scale_min"),
                scale_max=metrics_dict.get("scale_max"),
                has_zero_scale=metrics_dict.get("has_zero_scale", False),
                sensitivity_score=metrics_dict.get("sensitivity_score", 0.0),
            )
            result.layer_metrics[name] = metrics

        result.excluded_layers = data.get("excluded_layers", [])
        result.exclusion_reasons = data.get("exclusion_reasons", {})
        result.statistics = data.get("statistics", {})
        return result


class SensitivityAnalyzer:
    """
    Analyzes model layers to identify those sensitive to quantization.

    This class collects layer-wise metrics during calibration and applies
    configurable rules to determine which layers should be excluded.
    """

    # Default sensitivity thresholds
    DEFAULT_THRESHOLDS = {
        # Weight quantization error threshold (relative)
        "l1_error_weight_percentile": 95,  # Exclude layers above this percentile
        "l1_error_weight_multiplier": 2.0,  # Or above mean * multiplier
        # Activation error thresholds
        "l1_io_error_percentile": 95,
        "l1_io_error_multiplier": 2.0,
        # Scale thresholds
        "scale_outlier_multiplier": 10.0,  # Scale is outlier if > mean * multiplier
        # Minimum layers to exclude (as percentage)
        "min_exclude_percent": 0,  # Don't force minimum exclusions
        "max_exclude_percent": 20,  # Don't exclude more than this percentage
    }

    # Layer patterns that are always sensitive
    ALWAYS_SENSITIVE_PATTERNS = [
        "lm_head",  # Output projection
    ]

    # Layer patterns that are typically more sensitive
    TYPICALLY_SENSITIVE_PATTERNS = [
        "*embed*",  # Embedding layers
        "*norm*",  # Normalization layers
        "*gate*",  # Gate layers (excluding gate_proj which is MLP)
    ]

    def __init__(
        self,
        model: nn.Module,
        model_type: str | None = None,
        template: LLMTemplate | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize the sensitivity analyzer.

        Args:
            model: The PyTorch model to analyze
            model_type: Optional model type string
            template: Optional LLMTemplate for the model
            thresholds: Custom sensitivity thresholds
        """
        self.model = model
        self.model_type = model_type
        self.template = template
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

        self._metrics: dict[str, LayerMetrics] = {}
        self._debug_stats: dict[str, Any] = {}

    def _get_base_layer_name(self, quantizer_name: str) -> str:
        """Extract base layer name from quantizer name."""
        # Remove quantizer suffixes
        name = quantizer_name
        for suffix in [
            "._weight_quantizer",
            "._bias_quantizer",
            "._input_quantizer",
            "._output_quantizer",
        ]:
            name = name.replace(suffix, "")
        return name

    def _collect_weight_metrics(self) -> None:
        """Collect weight quantization metrics from the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                if "weight_quantizer" in name or "bias_quantizer" in name:
                    base_name = self._get_base_layer_name(name)

                    if base_name not in self._metrics:
                        self._metrics[base_name] = LayerMetrics(
                            name=base_name,
                            layer_type="linear",
                        )

                    # Check debug stats if available
                    if name in self._debug_stats:
                        stats = self._debug_stats[name]
                        self._metrics[base_name].l1_error = stats.get("l1_error")
                        self._metrics[base_name].shape = stats.get("shape")

    def _collect_scale_metrics(self) -> None:
        """Collect scale statistics from quantizers."""
        for name, module in self.model.named_modules():
            if isinstance(module, QuantMixin):
                for quantizer_attr in [
                    "_weight_quantizer",
                    "_input_quantizer",
                    "_output_quantizer",
                    "_bias_quantizer",
                ]:
                    quantizer = getattr(module, quantizer_attr, None)
                    if quantizer is not None and hasattr(quantizer, "scale"):
                        scale = quantizer.scale
                        if scale is not None:
                            base_name = name
                            if base_name in self._metrics:
                                self._metrics[base_name].scale_min = scale.min().item()
                                self._metrics[base_name].scale_max = scale.max().item()
                                self._metrics[base_name].has_zero_scale = scale.min().item() == 0.0

    def _compute_sensitivity_scores(self) -> None:
        """Compute sensitivity scores for each layer."""
        # Collect all metric values for normalization
        l1_errors = []
        l1_io_errors = []
        scales = []

        for metrics in self._metrics.values():
            if metrics.l1_error is not None:
                l1_errors.append(metrics.l1_error)
            if metrics.l1_io_error is not None:
                l1_io_errors.append(metrics.l1_io_error)
            if metrics.scale_max is not None:
                scales.append(metrics.scale_max)

        # Compute statistics
        l1_error_mean = np.mean(l1_errors) if l1_errors else 0
        l1_error_std = np.std(l1_errors) if l1_errors else 1
        l1_io_mean = np.mean(l1_io_errors) if l1_io_errors else 0
        scale_mean = np.mean(scales) if scales else 1

        # Score each layer
        for name, metrics in self._metrics.items():
            score = 0.0

            # Weight error contribution
            if metrics.l1_error is not None and l1_error_std > 0:
                z_score = (metrics.l1_error - l1_error_mean) / l1_error_std
                score += max(0, z_score) * 10  # Scale to 0-100 range typically

            # Activation error contribution
            if metrics.l1_io_error is not None and l1_io_mean > 0:
                rel_error = metrics.l1_io_error / l1_io_mean
                score += max(0, rel_error - 1) * 20

            # Scale outlier contribution
            if metrics.scale_max is not None and scale_mean > 0:
                scale_ratio = metrics.scale_max / scale_mean
                if scale_ratio > self.thresholds["scale_outlier_multiplier"]:
                    score += (scale_ratio - self.thresholds["scale_outlier_multiplier"]) * 5

            # Zero scale is very bad
            if metrics.has_zero_scale:
                score += 50

            # Pattern-based adjustments
            for pattern in self.ALWAYS_SENSITIVE_PATTERNS:
                if pattern in name:
                    score += 100
                    break

            for pattern in self.TYPICALLY_SENSITIVE_PATTERNS:
                if pattern.replace("*", "") in name.lower():
                    score += 10
                    break

            metrics.sensitivity_score = score

    def _identify_sensitive_layers(self) -> tuple[list[str], dict[str, str]]:
        """Identify layers that should be excluded based on rules."""
        excluded = []
        reasons = {}

        # Sort layers by sensitivity score
        sorted_layers = sorted(
            self._metrics.items(), key=lambda x: x[1].sensitivity_score, reverse=True
        )

        total_layers = len(sorted_layers)
        max_exclude = int(total_layers * self.thresholds["max_exclude_percent"] / 100)

        for name, metrics in sorted_layers:
            if len(excluded) >= max_exclude:
                break

            reason = None

            # Rule 1: Always sensitive patterns
            for pattern in self.ALWAYS_SENSITIVE_PATTERNS:
                if pattern in name:
                    reason = f"Always excluded (pattern: {pattern})"
                    break

            # Rule 2: High sensitivity score
            if reason is None and metrics.sensitivity_score >= 50:
                reason = f"High sensitivity score ({metrics.sensitivity_score:.2f})"

            # Rule 3: High weight quantization error
            if reason is None and metrics.l1_error is not None:
                l1_errors = [m.l1_error for m in self._metrics.values() if m.l1_error is not None]
                if l1_errors:
                    threshold = np.percentile(
                        l1_errors, self.thresholds["l1_error_weight_percentile"]
                    )
                    if metrics.l1_error > threshold:
                        reason = f"High weight quantization error ({metrics.l1_error:.4f})"

            # Rule 4: Zero scale
            if reason is None and metrics.has_zero_scale:
                reason = "Zero scale detected (may cause incorrect quantization)"

            # Rule 5: Template recommendations
            if reason is None and self.template:
                for pattern in self.template.exclude_layers_name:
                    if pattern in name or name in pattern:
                        reason = f"Recommended by template ({pattern})"
                        break

            if reason:
                excluded.append(name)
                reasons[name] = reason

        return excluded, reasons

    def analyze_from_calibration(
        self,
        debug_stats: dict[str, Any] | None = None,
        scale_stats: dict[str, Any] | None = None,
    ) -> SensitivityAnalysisResult:
        """
        Analyze sensitivity from calibration statistics.

        Args:
            debug_stats: Debug statistics from Quark's collect_quantization_statistics
            scale_stats: Scale statistics from Quark's check_scale_stats

        Returns:
            SensitivityAnalysisResult with layer metrics and exclusion recommendations
        """
        # Store debug stats for metric collection
        if debug_stats:
            self._debug_stats = debug_stats

        # Collect metrics
        self._collect_weight_metrics()
        self._collect_scale_metrics()

        # Process scale stats if provided
        if scale_stats:
            for layer_name, layer_stats in scale_stats.get("scale_stats", {}).items():
                if layer_name in self._metrics:
                    for quant_type in [
                        "_weight_quantizer",
                        "_input_quantizer",
                        "_output_quantizer",
                        "_bias_quantizer",
                    ]:
                        if quant_type in layer_stats:
                            qs = layer_stats[quant_type]
                            self._metrics[layer_name].scale_min = qs.get(
                                "scale_min_max", (None, None)
                            )[0]
                            self._metrics[layer_name].scale_max = qs.get(
                                "scale_min_max", (None, None)
                            )[1]
                            self._metrics[layer_name].has_zero_scale = qs.get(
                                "has_zero_scale", False
                            )

        # Compute sensitivity scores
        self._compute_sensitivity_scores()

        # Identify sensitive layers
        excluded, reasons = self._identify_sensitive_layers()

        # Compute summary statistics
        scores = [m.sensitivity_score for m in self._metrics.values()]
        l1_errors = [m.l1_error for m in self._metrics.values() if m.l1_error is not None]

        statistics = {
            "total_layers": len(self._metrics),
            "excluded_count": len(excluded),
            "excluded_percent": len(excluded) / len(self._metrics) * 100 if self._metrics else 0,
            "sensitivity_score_mean": np.mean(scores) if scores else 0,
            "sensitivity_score_std": np.std(scores) if scores else 0,
            "l1_error_mean": np.mean(l1_errors) if l1_errors else 0,
            "l1_error_std": np.std(l1_errors) if l1_errors else 0,
        }

        return SensitivityAnalysisResult(
            layer_metrics=self._metrics,
            excluded_layers=excluded,
            exclusion_reasons=reasons,
            statistics=statistics,
        )

    def analyze_from_json(self, debug_dir: str | Path) -> SensitivityAnalysisResult:
        """
        Analyze sensitivity from Quark debug JSON files.

        Args:
            debug_dir: Directory containing Quark debug output

        Returns:
            SensitivityAnalysisResult with layer metrics and exclusion recommendations
        """
        debug_dir = Path(debug_dir)
        debug_stats = {}

        # Load per-layer stats from JSON files
        for json_file in debug_dir.glob("*_stats.json"):
            try:
                with open(json_file) as f:
                    stats = json.load(f)

                # Extract layer name from filename
                layer_name = json_file.stem.replace("_stats", "")
                debug_stats[layer_name] = stats
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

        # Load scale stats if available
        scale_stats = None
        scale_file = Path("./debug_scale/scale_stats.json")
        if scale_file.exists():
            try:
                with open(scale_file) as f:
                    scale_stats = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load scale stats: {e}")

        return self.analyze_from_calibration(debug_stats, scale_stats)

    def get_top_sensitive_layers(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N most sensitive layers.

        Args:
            n: Number of layers to return

        Returns:
            List of (layer_name, sensitivity_score) tuples
        """
        sorted_layers = sorted(
            self._metrics.items(), key=lambda x: x[1].sensitivity_score, reverse=True
        )
        return [(name, m.sensitivity_score) for name, m in sorted_layers[:n]]

    def print_summary(self, result: SensitivityAnalysisResult) -> None:
        """Print a summary of the sensitivity analysis."""
        print("\n" + "=" * 60)
        print("LAYER SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal layers analyzed: {result.statistics.get('total_layers', 0)}")
        print(
            f"Layers recommended for exclusion: {result.statistics.get('excluded_count', 0)} "
            f"({result.statistics.get('excluded_percent', 0):.1f}%)"
        )

        print("\nSensitivity Score Statistics:")
        print(f"  Mean: {result.statistics.get('sensitivity_score_mean', 0):.2f}")
        print(f"  Std:  {result.statistics.get('sensitivity_score_std', 0):.2f}")

        if result.statistics.get("l1_error_mean"):
            print("\nL1 Error Statistics:")
            print(f"  Mean: {result.statistics.get('l1_error_mean', 0):.6f}")
            print(f"  Std:  {result.statistics.get('l1_error_std', 0):.6f}")

        print("\nTop 10 Most Sensitive Layers:")
        top_layers = self.get_top_sensitive_layers(10)
        for i, (name, score) in enumerate(top_layers, 1):
            excluded_marker = " [EXCLUDED]" if name in result.excluded_layers else ""
            print(f"  {i:2d}. {name}: {score:.2f}{excluded_marker}")

        print("\nExcluded Layers and Reasons:")
        for layer in result.excluded_layers[:20]:  # Show first 20
            reason = result.exclusion_reasons.get(layer, "Unknown")
            print(f"  - {layer}: {reason}")

        if len(result.excluded_layers) > 20:
            print(f"  ... and {len(result.excluded_layers) - 20} more")

        print("=" * 60)


def convert_exclusions_to_patterns(layer_names: list[str]) -> list[str]:
    """
    Convert layer names to wildcard patterns for exclusion.

    This function takes specific layer names and generates patterns that
    will match those layers during quantization.

    Args:
        layer_names: List of specific layer names

    Returns:
        List of wildcard patterns
    """
    patterns = []

    for name in layer_names:
        # Try to create a more general pattern
        # For example: model.layers.31.mlp.gate_proj -> model.layers.31.*

        # Check if it's a transformer layer
        layer_match = re.match(r"(model\.layers\.\d+)\..*", name)
        if layer_match:
            patterns.append(f"{layer_match.group(1)}.*")
            continue

        # Otherwise, use the exact name with wildcards
        patterns.append(f"*{name}*")

    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique_patterns.append(p)

    return unique_patterns
