"""
QAT Search Configuration.

Parses a YAML file into structured dataclasses for search space,
datasets, target criteria, and tuner settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
    """A single dataset with mixing ratio."""

    name: str
    ratio: float = 1.0
    subset: str | None = None
    split: str = "train"
    text_column: str = "text"


@dataclass
class SearchSpaceDimension:
    """One searchable hyperparameter dimension."""

    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    scale: str = "uniform"  # "uniform" or "log"


@dataclass
class DatasetRatioSearchSpec:
    """Specification for searching over dataset mix ratios."""

    datasets: list[str]
    min_ratio: float = 0.1


@dataclass
class TargetConfig:
    """Target metric and stopping criteria."""

    metric: str = "perplexity"
    mode: str = "min"
    threshold: float | None = None
    max_trials: int = 20
    max_total_time_seconds: int = 86400


@dataclass
class TrackingConfig:
    """Metric tracking backend configuration."""

    backends: list[str] = field(default_factory=lambda: ["tensorboard"])
    tensorboard_dir: str = ""
    wandb_project: str = "qat-search"
    wandb_entity: str = ""


@dataclass
class TunerConfig:
    """Configuration for the PBT tuner."""

    method: str = "pbt"
    population_size: int = 5
    exploit_interval: int = 1
    perturbation_factor: float = 0.2
    early_stopping_patience: int = 0
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


@dataclass
class QATSearchConfig:
    """Top-level configuration for QAT hyperparameter search."""

    # Model
    model_path: str
    output_dir: str
    trust_remote_code: bool = True

    # Search space: name -> SearchSpaceDimension
    search_space: dict[str, SearchSpaceDimension] = field(default_factory=dict)

    # Dataset ratio search (optional — if set, overrides fixed ratios)
    dataset_ratio_search: DatasetRatioSearchSpec | None = None

    # Fixed dataset config
    train_datasets: list[DatasetSpec] = field(default_factory=list)
    eval_dataset: DatasetSpec = field(
        default_factory=lambda: DatasetSpec(name="wikitext", split="test")
    )
    seq_len: int = 512
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Target
    target: TargetConfig = field(default_factory=TargetConfig)

    # Tuner config (replaces ray_config)
    tuner_config: TunerConfig = field(default_factory=TunerConfig)

    # Export
    export_weight_format: str = "real_quantized"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_search_space(raw: dict[str, Any]) -> dict[str, SearchSpaceDimension]:
    """Parse search_space section from YAML."""
    result = {}
    for name, dim_raw in raw.items():
        if name == "dataset_ratios":
            continue  # handled separately
        if not isinstance(dim_raw, dict):
            continue
        result[name] = SearchSpaceDimension(
            choices=dim_raw.get("choices"),
            min=float(dim_raw["min"]) if "min" in dim_raw else None,
            max=float(dim_raw["max"]) if "max" in dim_raw else None,
            scale=dim_raw.get("scale", "uniform"),
        )
    return result


def _parse_dataset_ratio_search(raw: dict[str, Any] | None) -> DatasetRatioSearchSpec | None:
    """Parse dataset_ratios from search_space section."""
    if raw is None:
        return None
    return DatasetRatioSearchSpec(
        datasets=raw.get("datasets", []),
        min_ratio=raw.get("min_ratio", 0.1),
    )


def _parse_datasets(raw: dict[str, Any]) -> tuple[list[DatasetSpec], DatasetSpec]:
    """Parse datasets section from YAML."""
    train_specs = []
    for ds in raw.get("train", []):
        train_specs.append(
            DatasetSpec(
                name=ds["name"],
                ratio=ds.get("ratio", 1.0),
                subset=ds.get("subset"),
                split=ds.get("split", "train"),
                text_column=ds.get("text_column", "text"),
            )
        )

    eval_raw = raw.get("eval", {})
    eval_spec = DatasetSpec(
        name=eval_raw.get("name", "wikitext"),
        subset=eval_raw.get("subset"),
        split=eval_raw.get("split", "test"),
        text_column=eval_raw.get("text_column", "text"),
    )

    return train_specs, eval_spec


def _parse_tuner(raw: dict[str, Any]) -> TunerConfig:
    """Parse tuner section from YAML."""
    tracking_raw = raw.get("tracking", {})
    tracking = TrackingConfig(
        backends=tracking_raw.get("backends", ["tensorboard"]),
        tensorboard_dir=tracking_raw.get("tensorboard_dir", ""),
        wandb_project=tracking_raw.get("wandb_project", "qat-search"),
        wandb_entity=tracking_raw.get("wandb_entity", ""),
    )
    return TunerConfig(
        method=raw.get("method", "pbt"),
        population_size=raw.get("population_size", 5),
        exploit_interval=raw.get("exploit_interval", 1),
        perturbation_factor=raw.get("perturbation_factor", 0.2),
        early_stopping_patience=raw.get("early_stopping_patience", 0),
        tracking=tracking,
    )


def _parse_target(raw: dict[str, Any]) -> TargetConfig:
    """Parse target section from YAML."""
    return TargetConfig(
        metric=raw.get("metric", "perplexity"),
        mode=raw.get("mode", "min"),
        threshold=raw.get("threshold"),
        max_trials=raw.get("max_trials", 20),
        max_total_time_seconds=raw.get("max_total_time_seconds", 86400),
    )


def load_search_config(path: str | Path) -> QATSearchConfig:
    """Load and parse a QAT search YAML config file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model = raw.get("model", {})
    space_raw = raw.get("search_space", {})
    datasets_raw = raw.get("datasets", {})
    target_raw = raw.get("target", {})
    tuner_raw = raw.get("tuner", {})

    search_space = _parse_search_space(space_raw)
    dataset_ratio_search = _parse_dataset_ratio_search(space_raw.get("dataset_ratios"))
    train_datasets, eval_dataset = _parse_datasets(datasets_raw)
    target = _parse_target(target_raw)
    tuner_config = _parse_tuner(tuner_raw)

    model_path = model.get("model_path")
    if not model_path:
        raise ValueError(
            "The 'model_path' field is required under the 'model' section in the config."
        )

    return QATSearchConfig(
        model_path=model_path,
        output_dir=model.get("output_dir", "./qat_output"),
        trust_remote_code=model.get("trust_remote_code", True),
        search_space=search_space,
        dataset_ratio_search=dataset_ratio_search,
        train_datasets=train_datasets,
        eval_dataset=eval_dataset,
        seq_len=datasets_raw.get("seq_len", 512),
        max_train_samples=datasets_raw.get("max_train_samples"),
        max_eval_samples=datasets_raw.get("max_eval_samples"),
        target=target,
        tuner_config=tuner_config,
        export_weight_format=model.get("export_weight_format", "real_quantized"),
    )
