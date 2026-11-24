"""漂移监测：PSI（群体稳定性指数）+ KS 双样本检验。

选择 PSI 作主指标的理由：工业场景中标签往往延迟可得，PSI 只需特征分布，
可在无标签情况下先报警；KS 用于交叉验证，避免分箱方式带来的假阳性。
"""

from __future__ import annotations

import numpy as np


def _bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, qs)
    edges = np.unique(edges)
    if len(edges) < 3:  # 退化分布
        m, s = float(reference.mean()), float(reference.std() + 1e-8)
        edges = np.array([m - 3 * s, m, m + 3 * s])
    return edges


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI = sum((实际占比-期望占比)*ln(实际/期望))。>0.1 关注，>0.25 显著。"""
    edges = _bin_edges(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """两样本 KS 统计量（经验分布函数最大差距）。"""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def monitor(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    psi_threshold: float = 0.10,
) -> dict:
    """返回每列 PSI/KS、整体告警等级与 Top 漂移特征。"""
    results = []
    for j, name in enumerate(feature_names):
        p = psi(reference[:, j], current[:, j])
        k = ks_statistic(reference[:, j], current[:, j])
        results.append({"feature": name, "psi": p, "ks": k})

    results.sort(key=lambda r: r["psi"], reverse=True)
    max_psi = results[0]["psi"] if results else 0.0
    if max_psi >= 0.25:
        level = "SEVERE"
    elif max_psi >= psi_threshold:
        level = "WARN"
    else:
        level = "OK"

    return {
        "level": level,
        "max_psi": max_psi,
        "features": results,
        "top_k": [r["feature"] for r in results[:2]],
    }
