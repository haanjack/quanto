"""
QAT Search Configuration.

Parses a single YAML file into structured dataclasses for search space,
datasets, target criteria, and Ray Tune passthrough settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    eval_dataset: DatasetSpec = field(default_factory=lambda: DatasetSpec(name="wikitext", split="test"))
    seq_len: int = 512
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Target
    target: TargetConfig = field(default_factory=TargetConfig)

    # Ray Tune passthrough — raw dict from YAML
    ray_config: dict[str, Any] = field(default_factory=dict)

    # Export
    export_weight_format: str = "real_quantized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "trust_remote_code": self.trust_remote_code,
            "search_space": {
                k: {"choices": v.choices, "min": v.min, "max": v.max, "scale": v.scale}
                for k, v in self.search_space.items()
            },
            "train_datasets": [{"name": d.name, "ratio": d.ratio} for d in self.train_datasets],
            "eval_dataset": {"name": self.eval_dataset.name, "split": self.eval_dataset.split},
            "seq_len": self.seq_len,
            "target": {
                "metric": self.target.metric,
                "mode": self.target.mode,
                "threshold": self.target.threshold,
                "max_trials": self.target.max_trials,
            },
        }


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
            min=dim_raw.get("min"),
            max=dim_raw.get("max"),
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
        train_specs.append(DatasetSpec(
            name=ds["name"],
            ratio=ds.get("ratio", 1.0),
            split=ds.get("split", "train"),
            text_column=ds.get("text_column", "text"),
        ))

    eval_raw = raw.get("eval", {})
    eval_spec = DatasetSpec(
        name=eval_raw.get("name", "wikitext"),
        split=eval_raw.get("split", "test"),
        text_column=eval_raw.get("text_column", "text"),
    )

    return train_specs, eval_spec


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
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = raw.get("model", {})
    space_raw = raw.get("search_space", {})
    datasets_raw = raw.get("datasets", {})
    target_raw = raw.get("target", {})
    ray_raw = raw.get("ray", {})

    search_space = _parse_search_space(space_raw)
    dataset_ratio_search = _parse_dataset_ratio_search(space_raw.get("dataset_ratios"))
    train_datasets, eval_dataset = _parse_datasets(datasets_raw)
    target = _parse_target(target_raw)

    return QATSearchConfig(
        model_path=model["model_path"],
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
        ray_config=ray_raw,
        export_weight_format=model.get("export_weight_format", "real_quantized"),
    )
