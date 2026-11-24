"""评估：AUC / Accuracy / ECE，以及"时间切分 + 双切片"稳健评估。

双切片是赛题"长期稳定"要求的具体落地：
    - stable 切片：漂移前的历史分布，用于检测"新模型是否损坏旧能力"；
    - recent 切片：漂移后的最新分布，用于检测"是否真的适应了新工况"。
只有两个切片同时不退化的模型才允许上线。
"""

from __future__ import annotations

import numpy as np


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 处理并列
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float(
        (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(np.mean(pred == np.asarray(y_true)))


def ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差：上线模型的可信度指标（工业场景比 AUC 更贴近决策）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_score, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(np.mean(y_score[m]))
        acc = float(np.mean(y_true[m]))
        total += abs(conf - acc) * m.mean()
    return float(total)


def evaluate_slices(
    model,
    X: np.ndarray,
    y: np.ndarray,
    slices: dict[str, slice],
) -> dict:
    out = {}
    for name, sl in slices.items():
        Xs, ys = X[sl], y[sl]
        if len(ys) == 0:
            continue
        score = model.predict_proba(Xs)
        out[name] = {
            "auc": auc(ys, score),
            "acc": accuracy(ys, score),
            "ece": ece(ys, score),
            "n": int(len(ys)),
        }
    return out
