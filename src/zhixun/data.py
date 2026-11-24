"""合成工业数据与漂移注入。

真实赛题数据由命题企业提供；本模块用于在没有企业数据时跑通闭环，
并可**可控地注入漂移**，从而复现"监测->归因->优化"的完整链路。

场景：注塑机工艺参数 -> 良率（二分类）。
特征含义：
    0 料筒温度(z)      1 注射压力(MPa)    2 保压时间(s)
    3 冷却时间(s)      4 原料含水率(%)    5 车间湿度(%)
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "料筒温度",
    "注射压力",
    "保压时间",
    "冷却时间",
    "原料含水率",
    "车间湿度",
]


def _logit(X: np.ndarray) -> np.ndarray:
    """真实的工业过程响应面（含交互项与饱和效应）。"""
    z = (
        0.9 * X[:, 0]
        - 1.1 * np.abs(X[:, 1] - 0.3)
        + 0.8 * X[:, 2]
        - 0.6 * np.abs(X[:, 3])
        - 1.4 * X[:, 4] * X[:, 5]
        + 0.5 * X[:, 0] * X[:, 4]
    )
    return z


def make_dataset(
    n_samples: int = 4000,
    n_features: int = 6,
    drift_start_ratio: float = 0.6,
    drift_strength: float = 1.0,
    label_noise: float = 0.05,
    seed: int = 7,
) -> dict:
    """返回 dict：X, y, t, drift_start, feature_names。

    漂移设计（模拟真实产线换料/换模）：
        - 原料含水率(4) 均值 +0.8*strength，方差放大
        - 车间湿度(5)   均值 -0.5*strength
        - 二者交互项对良率的影响被削弱 30%，即"旧模型学到的关系失效"
    """
    rng = np.random.default_rng(seed)
    n_features = min(n_features, len(FEATURE_NAMES))
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    drift_start = int(n_samples * drift_start_ratio)

    X_after = X.copy()
    if n_features > 4:
        X_after[drift_start:, 4] = (
            X_after[drift_start:, 4] * 1.4 + 0.8 * drift_strength
        )
    if n_features > 5:
        X_after[drift_start:, 5] = X_after[drift_start:, 5] - 0.5 * drift_strength

    z = _logit(X_after)
    # 漂移后交互项失效：削弱 X4*X5 的贡献
    if n_features > 5:
        z[drift_start:] -= 0.4 * drift_strength * X_after[drift_start:, 4] * X_after[drift_start:, 5]

    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n_samples) < p).astype(np.int64)
    flip = rng.random(n_samples) < label_noise
    y[flip] ^= 1

    return {
        "X": X_after,
        "y": y,
        "t": np.arange(n_samples),
        "drift_start": drift_start,
        "feature_names": FEATURE_NAMES[:n_features],
    }


def time_split(n: int, train_ratio: float = 0.55, val_ratio: float = 0.15) -> dict:
    """按时间切分，禁止未来信息泄漏。"""
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_val),
        "test": slice(n_val, n),
    }
