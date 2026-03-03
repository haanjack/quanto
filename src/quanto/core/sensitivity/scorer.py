"""
Sensitivity Scorer for Quantization Impact Analysis.

Measures the deviation between baseline (FP16) and quantized activations
to determine layer sensitivity to quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class SensitivityMetric(Enum):
    """Metrics for measuring sensitivity."""
    MSE = "mse"                    # Mean Squared Error
    MAE = "mae"                    # Mean Absolute Error
    COSINE = "cosine"              # 1 - Cosine Similarity
    KL_DIVERGENCE = "kl"           # KL Divergence (for distributions)
    RELATIVE_NORM = "relative"     # Relative L2 norm change


@dataclass
class SensitivityScore:
    """Sensitivity score for a single layer."""
    layer_name: str
    layer_idx: int
    score: float
    metric: SensitivityMetric
    baseline_norm: float
    deviation_norm: float

    def __lt__(self, other: SensitivityScore) -> bool:
        return self.score < other.score

    def __repr__(self) -> str:
        return f"SensitivityScore({self.layer_name}: {self.score:.6f})"


class SensitivityScorer:
    """
    Calculate sensitivity scores for quantized layers.

    Compares quantized layer outputs against baseline (FP16) outputs
    to measure how sensitive each layer is to quantization.
    """

    def __init__(
        self,
        metric: SensitivityMetric = SensitivityMetric.RELATIVE_NORM,
        aggregation: str = "mean",  # "mean", "max", "sum"
    ):
        """
        Initialize the sensitivity scorer.

        Args:
            metric: The metric to use for comparing activations
            aggregation: How to aggregate scores across multiple samples
        """
        self.metric = metric
        self.aggregation = aggregation
        self._scores: dict[str, list[float]] = {}

    def compute_score(
        self,
        baseline: torch.Tensor,
        quantized: torch.Tensor,
    ) -> float:
        """
        Compute sensitivity score between baseline and quantized tensors.

        Args:
            baseline: The baseline (FP16) activation tensor
            quantized: The quantized activation tensor

        Returns:
            Sensitivity score (higher = more sensitive)
        """
        if self.metric == SensitivityMetric.MSE:
            return F.mse_loss(baseline, quantized).item()

        elif self.metric == SensitivityMetric.MAE:
            return F.l1_loss(baseline, quantized).item()

        elif self.metric == SensitivityMetric.COSINE:
            # 1 - cosine similarity (0 = identical, 2 = opposite)
            cos_sim = F.cosine_similarity(
                baseline.flatten().unsqueeze(0),
                quantized.flatten().unsqueeze(0),
            )
            return (1.0 - cos_sim.item())

        elif self.metric == SensitivityMetric.KL_DIVERGENCE:
            # For KL divergence, treat activations as probability distributions
            p = F.softmax(baseline.flatten(), dim=0)
            q = F.softmax(quantized.flatten(), dim=0)
            return F.kl_div(
                q.log(),
                p,
                reduction="sum",
            ).item()

        elif self.metric == SensitivityMetric.RELATIVE_NORM:
            # Relative L2 norm change: ||x - x_hat|| / ||x||
            diff_norm = torch.norm(baseline - quantized).item()
            baseline_norm = torch.norm(baseline).item()

            if baseline_norm < 1e-8:
                return 0.0 if diff_norm < 1e-8 else float("inf")

            return diff_norm / baseline_norm

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def record_layer_score(
        self,
        layer_name: str,
        layer_idx: int,
        baseline_output: torch.Tensor,
        quantized_output: torch.Tensor,
    ) -> SensitivityScore:
        """
        Record and compute sensitivity score for a layer.

        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
            baseline_output: Baseline (FP16) layer output
            quantized_output: Quantized layer output

        Returns:
            SensitivityScore object with the computed score
        """
        score = self.compute_score(baseline_output, quantized_output)

        # Store for aggregation
        if layer_name not in self._scores:
            self._scores[layer_name] = []
        self._scores[layer_name].append(score)

        baseline_norm = torch.norm(baseline_output).item()
        deviation_norm = torch.norm(baseline_output - quantized_output).item()

        return SensitivityScore(
            layer_name=layer_name,
            layer_idx=layer_idx,
            score=score,
            metric=self.metric,
            baseline_norm=baseline_norm,
            deviation_norm=deviation_norm,
        )

    def get_aggregated_scores(self) -> list[SensitivityScore]:
        """
        Get aggregated sensitivity scores for all layers.

        Returns:
            List of SensitivityScore objects, sorted by score (highest first)
        """
        aggregated = []

        for layer_name, scores in self._scores.items():
            if self.aggregation == "mean":
                final_score = sum(scores) / len(scores)
            elif self.aggregation == "max":
                final_score = max(scores)
            elif self.aggregation == "sum":
                final_score = sum(scores)
            else:
                final_score = sum(scores) / len(scores)

            # Extract layer index from name (e.g., "model.layers.5.mlp" -> 5)
            layer_idx = self._extract_layer_idx(layer_name)

            aggregated.append(SensitivityScore(
                layer_name=layer_name,
                layer_idx=layer_idx,
                score=final_score,
                metric=self.metric,
                baseline_norm=0.0,  # Not available after aggregation
                deviation_norm=0.0,
            ))

        # Sort by score (highest = most sensitive)
        aggregated.sort(reverse=True)
        return aggregated

    def _extract_layer_idx(self, layer_name: str) -> int:
        """Extract layer index from layer name."""
        import re
        match = re.search(r"layers\.(\d+)", layer_name)
        if match:
            return int(match.group(1))
        return -1

    def get_layers_above_threshold(
        self,
        threshold: float,
    ) -> list[str]:
        """
        Get layers with sensitivity above the given threshold.

        Args:
            threshold: Sensitivity threshold

        Returns:
            List of layer names that exceed the threshold
        """
        scores = self.get_aggregated_scores()
        return [s.layer_name for s in scores if s.score > threshold]

    def clear(self) -> None:
        """Clear all recorded scores."""
        self._scores.clear()

    def get_summary(self) -> str:
        """Get a summary of sensitivity scores."""
        scores = self.get_aggregated_scores()

        if not scores:
            return "No sensitivity scores recorded"

        lines = ["Sensitivity Scores (highest first):"]
        for s in scores[:10]:  # Top 10
            lines.append(f"  {s.layer_name}: {s.score:.6f}")

        if len(scores) > 10:
            lines.append(f"  ... and {len(scores) - 10} more layers")

        return "\n".join(lines)
