"""
Population member state management for PBT.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from .distributed import is_rank0

logger = logging.getLogger(__name__)


@dataclass
class PopulationMember:
    """Tracks one member of the PBT population."""

    member_id: int
    hyperparams: dict[str, Any]
    checkpoint_path: str
    metric_history: list[float] = field(default_factory=list)
    best_metric: float = float("inf")
    total_epochs_trained: int = 0
    finished: bool = False

    def record_metric(self, value: float, mode: str = "min") -> bool:
        """Record a metric value. Returns True if best improved."""
        self.metric_history.append(value)
        improved = False
        if mode == "min" and value < self.best_metric or mode == "max" and value > self.best_metric:
            self.best_metric = value
            improved = True
        return improved

    def to_dict(self) -> dict:
        return {
            "member_id": self.member_id,
            "hyperparams": self.hyperparams,
            "checkpoint_path": self.checkpoint_path,
            "metric_history": self.metric_history,
            "best_metric": self.best_metric,
            "total_epochs_trained": self.total_epochs_trained,
            "finished": self.finished,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PopulationMember:
        return cls(
            member_id=d["member_id"],
            hyperparams=d["hyperparams"],
            checkpoint_path=d["checkpoint_path"],
            metric_history=d.get("metric_history", []),
            best_metric=d.get("best_metric", float("inf")),
            total_epochs_trained=d.get("total_epochs_trained", 0),
            finished=d.get("finished", False),
        )


def clone_checkpoint(src_member: PopulationMember, dst_member: PopulationMember) -> None:
    """Copy checkpoint from donor to recipient for PBT exploit.

    Only copies files on disk, not GPU memory.
    The recipient loads the checkpoint on its next train_segment call.
    Only rank 0 should call this.
    """
    if not is_rank0():
        return
    if os.path.exists(dst_member.checkpoint_path):
        shutil.rmtree(dst_member.checkpoint_path)

    if os.path.exists(src_member.checkpoint_path):
        shutil.copytree(src_member.checkpoint_path, dst_member.checkpoint_path)
    else:
        logger.warning(
            f"No checkpoint to clone from member {src_member.member_id}, "
            f"member {dst_member.member_id} will start fresh"
        )


def save_population_state(
    population: list[PopulationMember],
    round_num: int,
    state_path: str,
) -> None:
    """Save population state for resume. Only rank 0 writes."""
    if not is_rank0():
        return
    state = {
        "round": round_num,
        "population": [m.to_dict() for m in population],
    }
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_population_state(
    state_path: str,
) -> tuple[list[PopulationMember], int]:
    """Load population state from disk. Returns (population, round_num)."""
    with open(state_path) as f:
        state = json.load(f)
    population = [PopulationMember.from_dict(d) for d in state["population"]]
    round_num = state.get("round", 0)
    return population, round_num
