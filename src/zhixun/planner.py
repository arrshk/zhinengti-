"""实验优化：在预算约束下生成候选配置，并按历史证据排序（训练师策略）。

这是"训练师模型"的轻量替代：用 UCB1 多臂老虎机把"配置族 -> 历史收益"
积累成策略，避免每轮都从零网格搜索，符合赛题"预算受限下自主判断"的要求。
若队伍实现了 LLM 训练师，可将 Planner.select() 换成模型推理，接口不变。
"""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


DEFAULT_SPACE = {
    "lr": [0.05, 0.15, 0.3],
    "l2": [1e-4, 1e-3, 1e-2],
    "epochs": [80, 160],
    "recent_weight": [1.0, 3.0, 8.0],
    "drop_feature": [None],  # 由诊断结果动态填充
}


def build_space(search_space: dict, drift_features: list[str], feature_names: list[str]) -> list[dict]:
    """展开搜索空间，并把"丢弃漂移特征"作为一档候选。"""
    space = {**DEFAULT_SPACE, **search_space}
    drop_opts = [None]
    for name in drift_features[:1]:
        if name in feature_names:
            drop_opts.append(name)

    keys = ["lr", "l2", "epochs", "recent_weight", "drop_feature"]
    combos = list(itertools.product(*[space.get(k, [None]) for k in keys]))

    candidates = []
    for lr, l2, epochs, rw, drop in combos:
        mask = None
        if drop is not None:
            mask = [i for i, n in enumerate(feature_names) if n != drop]
        candidates.append(
            {
                "lr": float(lr),
                "l2": float(l2),
                "epochs": int(epochs),
                "recent_weight": float(rw),
                "feature_mask": mask,
                "drop_feature": drop,
            }
        )
    return candidates


class UCBPlanner:
    """按历史实验收益对候选配置打分并排序。"""

    def __init__(self, history: Iterable[dict] | None = None, c: float = 0.15):
        self.c = c
        self.stats: dict[str, list[float]] = {}
        for exp in history or []:
            self.update(exp)

    @staticmethod
    def key(cfg: dict) -> str:
        return (
            f"lr={cfg['lr']},l2={cfg['l2']},ep={cfg['epochs']},"
            f"rw={cfg['recent_weight']},drop={cfg.get('drop_feature')}"
        )

    def update(self, exp: dict) -> None:
        cfg = exp.get("config", {})
        if "lr" not in cfg or "recent_weight" not in cfg:
            return  # 基线等非搜索实验不进入策略统计
        k = self.key(cfg)
        recent_auc = float(exp.get("metrics", {}).get("recent", {}).get("auc", 0.0))
        self.stats.setdefault(k, []).append(recent_auc)

    def score(self, cfg: dict, total_pulls: int) -> float:
        k = self.key(cfg)
        pulls = self.stats.get(k, [])
        if not pulls:
            return 1.0  # 未探索的配置优先
        mean = float(np.mean(pulls))
        bonus = self.c * np.sqrt(np.log(max(total_pulls, 1)) / len(pulls))
        return mean + bonus

    def select(self, candidates: list[dict], k: int = 3, **_) -> list[dict]:
        total = sum(len(v) for v in self.stats.values())
        ranked = sorted(candidates, key=lambda c: self.score(c, total), reverse=True)
        # 去重：同一 (drop_feature, recent_weight) 只留一个最优
        seen, out = set(), []
        for cfg in ranked:
            sig = (cfg.get("drop_feature"), cfg["recent_weight"], cfg["epochs"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cfg)
            if len(out) >= k:
                break
        return out


def make_planner(kind: str = "ucb", history: list[dict] | None = None, **kwargs):
    """按名字构造规划器：ucb（默认，可离线）或 qwen（需 PyTRIO/OpenAI）。"""
    if kind == "qwen":
        from .qwen_planner import QwenPlanner

        return QwenPlanner(fallback=UCBPlanner(history), **kwargs)
    return UCBPlanner(history)
