"""配置加载与配置哈希。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """稳定序列化：键排序、无多余空格，保证同一配置得到同一哈希。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


@dataclass
class DataConfig:
    n_samples: int = 4000
    n_features: int = 6
    drift_start_ratio: float = 0.60
    drift_strength: float = 1.0
    label_noise: float = 0.05
    seed: int = 7


@dataclass
class BudgetConfig:
    compute_minutes: float = 60.0
    annotation_items: int = 500
    minutes_per_train: float = 2.5
    items_per_label_query: int = 50


@dataclass
class GateConfig:
    min_auc_gain: float = 0.005
    max_regression: float = 0.02
    max_rounds: int = 3
    max_candidates_per_round: int = 3


@dataclass
class DriftConfig:
    psi_threshold: float = 0.10
    psi_alert: float = 0.25
    window: int = 800
    reference: int = 1200


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    search_space: dict = field(default_factory=dict)
    output_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(
            data=DataConfig(**raw.get("data", {})),
            budget=BudgetConfig(**raw.get("budget", {})),
            gate=GateConfig(**raw.get("gate", {})),
            drift=DriftConfig(**raw.get("drift", {})),
            search_space=raw.get("search_space", {}),
            output_dir=raw.get("output_dir", "runs"),
        )
        return cfg

    def to_dict(self) -> dict:
        return {
            "data": self.data.__dict__,
            "budget": self.budget.__dict__,
            "gate": self.gate.__dict__,
            "drift": self.drift.__dict__,
            "search_space": self.search_space,
            "output_dir": self.output_dir,
        }

    def hash(self) -> str:
        return config_hash(self.to_dict())
