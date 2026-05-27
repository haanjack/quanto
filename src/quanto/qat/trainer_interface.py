"""
Trainer-agnostic interface for QAT hyperparameter tuning.

Defines the protocol any training backend must satisfy, plus metric
tracking sinks (TensorBoard, W&B) that plug into the MetricCallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------


class MetricSink(Protocol):
    """Protocol for metric tracking backends."""

    def log(self, metrics: dict[str, float], step: int | None = None) -> None: ...


class TensorBoardSink:
    """Writes scalars to TensorBoard events files."""

    def __init__(self, log_dir: str, run_name: str):
        try:
            from torch.utils.tensorboard.writer import SummaryWriter
        except ImportError as err:
            raise ImportError("TensorBoard not installed. Run: pip install tensorboard") from err
        self._writer = SummaryWriter(log_dir=log_dir, comment=run_name)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step or 0)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandbSink:
    """Logs to a W&B run (one run per population member)."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict | None = None,
        entity: str | None = None,
    ):
        try:
            import wandb
        except ImportError as err:
            raise ImportError("wandb not installed. Run: pip install wandb") from err
        self._run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            entity=entity or None,
            reinit=True,
        )

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        log_dict = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if step is not None:
            log_dict["step"] = step
        self._run.log(log_dict)

    def finish(self) -> None:
        self._run.finish()


class MetricCallback:
    """Generic metric collector injected by the tuner into the trainer.

    Delegates to registered MetricSink backends (TensorBoard, W&B, etc).
    """

    def __init__(self, sinks: list[Any] | None = None):
        self.sinks: list[Any] = sinks or []
        self.metrics_history: list[dict[str, float]] = []
        self._current_metrics: dict[str, float] = {}
        self._step: int = 0

    def report(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_history.append(metrics.copy())
        self._current_metrics = metrics
        if step is not None:
            self._step = step
        else:
            self._step += 1
        for sink in self.sinks:
            try:
                sink.log(metrics, self._step)
            except Exception as e:
                logger.warning(f"MetricSink log failed: {e}")

    def last(self) -> dict[str, float]:
        return self._current_metrics

    @property
    def step(self) -> int:
        return self._step


def build_sinks(
    member_id: int,
    tracking_config: Any,
    output_dir: str,
    hyperparams: dict | None = None,
) -> list[Any]:
    """Build metric sinks for a population member based on tracking config."""
    sinks = []
    for backend in tracking_config.backends:
        if backend == "tensorboard":
            import os

            tb_dir = tracking_config.tensorboard_dir or os.path.join(output_dir, "tb_logs")
            sinks.append(TensorBoardSink(log_dir=tb_dir, run_name=f"member_{member_id}"))
        elif backend == "wandb":
            sinks.append(
                WandbSink(
                    project=tracking_config.wandb_project,
                    run_name=f"pbt-member-{member_id}",
                    config=hyperparams,
                    entity=tracking_config.wandb_entity or None,
                )
            )
        else:
            logger.warning(f"Unknown tracking backend: {backend}")
    return sinks


def close_sinks(sinks: list[Any]) -> None:
    """Close all sinks (e.g., W&B needs explicit finish)."""
    for sink in sinks:
        try:
            if hasattr(sink, "finish"):
                sink.finish()
            elif hasattr(sink, "close"):
                sink.close()
        except Exception as e:
            logger.warning(f"Sink close failed: {e}")


# ---------------------------------------------------------------------------
# Trainer protocol
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Returned after a single training segment."""

    metrics: dict[str, float]
    epoch: int
    finished: bool


class QATTrainer(Protocol):
    """Protocol (structural typing) for a QAT trainer backend.

    Any trainer (HF, Megatron, custom) implements these methods.
    The PBT tuner calls only these.
    """

    def initialize(self, config: dict) -> None:
        """Load model, apply quantization, build datasets."""
        ...

    def train_segment(
        self,
        hyperparams: dict[str, Any],
        num_epochs: int,
        metric_callback: MetricCallback,
        resume_from: str | None = None,
    ) -> TrainResult:
        """Train for num_epochs, reporting metrics via callback."""
        ...

    def evaluate(self) -> dict[str, float]:
        """Run evaluation and return metrics dict."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save model + optimizer state for PBT exploit."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimizer state."""
        ...

    def cleanup(self) -> None:
        """Release GPU memory."""
        ...
