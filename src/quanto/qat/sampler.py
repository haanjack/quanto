"""Population sampling with random hyperparameter generation."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any

from .config import DatasetRatioSearchSpec, SearchSpaceDimension
from .distributed import is_rank0
from .population import PopulationMember

logger = logging.getLogger(__name__)


def _sample_random(
    search_space: dict[str, SearchSpaceDimension],
    dataset_ratio_search: DatasetRatioSearchSpec | None = None,
) -> dict[str, Any]:
    """Sample a config using random sampling."""
    config: dict[str, Any] = {}
    for name, dim in search_space.items():
        if dim.choices is not None:
            config[name] = random.choice(dim.choices)
        elif dim.min is not None and dim.max is not None:
            if dim.scale == "log":
                log_min, log_max = math.log(dim.min), math.log(dim.max)
                config[name] = math.exp(random.uniform(log_min, log_max))
            else:
                config[name] = random.uniform(dim.min, dim.max)
        else:
            raise ValueError(f"Invalid dimension {name}: need choices or min/max")

    if dataset_ratio_search is not None:
        n = len(dataset_ratio_search.datasets)
        for i in range(n):
            config[f"ds_ratio_{i}"] = random.uniform(0.0, 1.0)

    return config


def sample_initial_population(
    search_space: dict[str, SearchSpaceDimension],
    population_size: int,
    output_dir: str,
    dataset_ratio_search: DatasetRatioSearchSpec | None = None,
) -> list[PopulationMember]:
    """Sample the initial PBT population using random hyperparameter generation."""
    logger.info("Using random sampling for initial population")

    members = []
    if is_rank0():
        pop_dir = os.path.join(output_dir, "population")
        os.makedirs(pop_dir, exist_ok=True)

    for i in range(population_size):
        config = _sample_random(search_space, dataset_ratio_search)

        checkpoint_dir = os.path.join(output_dir, "population", f"member_{i}")
        if is_rank0():
            os.makedirs(checkpoint_dir, exist_ok=True)
            config_path = os.path.join(checkpoint_dir, "hyperparams.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        members.append(
            PopulationMember(
                member_id=i,
                hyperparams=config,
                checkpoint_path=checkpoint_dir,
            )
        )
        logger.info(f"Member {i}: {config}")

    return members
