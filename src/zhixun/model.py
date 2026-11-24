"""轻量逻辑回归（纯 numpy 实现，避免重型依赖，便于评审一键复现）。

之所以不用深度学习框架：赛题考察的是"训练与调参引擎"而非模型规模，
在算力预算约束下，轻量模型能让每一轮实验在秒级完成，从而跑出多轮 RSI 证据。
替换为企业模型时，只需保持 fit/predict_proba/save/load 四个接口。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LogisticModel:
    def __init__(
        self,
        lr: float = 0.15,
        l2: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 256,
        seed: int = 0,
        feature_mask: list[int] | None = None,
        recent_weight: float = 1.0,
    ):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.feature_mask = feature_mask
        self.recent_weight = float(recent_weight)

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    # ---------- 内部 ----------
    def _select(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mask is None:
            return X
        return X[:, self.feature_mask]

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    # ---------- 训练 ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        X = self._select(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)

        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        rng = np.random.default_rng(self.seed)

        # 近期样本加权：漂移后"用新数据说话"，是受控递归自改进的关键一环
        w_sample = np.linspace(1.0, self.recent_weight, n)

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                xb, yb, wb = Xs[batch], y[batch], w_sample[batch]
                p = 1.0 / (1.0 + np.exp(-(xb @ self.w + self.b)))
                err = (p - yb) * wb
                grad_w = xb.T @ err / len(batch) + self.l2 * self.w
                grad_b = err.mean()
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(self._select(np.asarray(X, dtype=float)))
        return 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))

    # ---------- 持久化 ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w=self.w,
            b=np.array([self.b]),
            mu=self.mu,
            sigma=self.sigma,
        )
        meta = {
            "lr": self.lr,
            "l2": self.l2,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "feature_mask": self.feature_mask,
            "recent_weight": self.recent_weight,
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        obj = cls(**meta)
        z = np.load(path)
        obj.w = z["w"]
        obj.b = float(z["b"][0])
        obj.mu = z["mu"]
        obj.sigma = z["sigma"]
        return obj
