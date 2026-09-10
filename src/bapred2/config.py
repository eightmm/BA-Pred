from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GraphConfig:
    pocket_cutoff: float = 8.0
    interface_cutoff: float = 5.0
    protein_spatial_cutoff: float = 5.0
    ligand_spatial_cutoff: float = 4.5
    max_spatial_neighbors: int = 32
    distance_rbf_dim: int = 16
    rwpe_dim: int = 20


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    prelude_layers: int = 1
    dropout: float = 0.1
    layerscale_init: float = 0.1
    use_endpoint_context: bool = True
    train_recycles: list[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    train_recycle_probs: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.20, 0.20, 0.10])
    eval_recycles: int = 6


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    loss: str = "huber"
    huber_delta: float = 1.0
    amp: bool = True
    early_stop_patience: int = 20


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _update_dataclass(cfg, values)
