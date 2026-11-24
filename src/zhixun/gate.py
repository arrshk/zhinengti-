"""上线门禁与回滚决策。

规则（可配置，默认保守）：
    上线：recent.auc >= base.recent.auc + min_auc_gain
          且 stable.auc >= base.stable.auc - max_regression
    回滚：上线后监控窗口内 recent.auc 跌破 base.recent.auc - max_regression
          或 ECE 恶化超过 0.05，即回滚到上一版本并记为一次失败实验。
"""

from __future__ import annotations


def decide(
    base: dict,
    cand: dict,
    min_auc_gain: float = 0.005,
    max_regression: float = 0.02,
) -> tuple[str, str]:
    """返回 (decision, reason)，decision ∈ {deploy, reject}。"""
    base_recent = base.get("recent", {}).get("auc", 0.0)
    base_stable = base.get("stable", {}).get("auc", 0.0)
    cand_recent = cand.get("recent", {}).get("auc", 0.0)
    cand_stable = cand.get("stable", {}).get("auc", 0.0)

    if cand_recent < base_recent + min_auc_gain:
        return "reject", (
            f"recent AUC {cand_recent:.4f} 未超过基线 {base_recent:.4f}"
            f"+{min_auc_gain:.4f}，收益不足"
        )
    if cand_stable < base_stable - max_regression:
        return "reject", (
            f"stable AUC {cand_stable:.4f} 相对基线 {base_stable:.4f} "
            f"回退超过 {max_regression:.4f}，存在灾难性遗忘风险"
        )
    return "deploy", (
        f"recent AUC +{cand_recent - base_recent:.4f}，stable AUC "
        f"{cand_stable - base_stable:+.4f}，通过双切片门禁"
    )


def should_rollback(
    deployed: dict,
    monitored: dict,
    max_regression: float = 0.02,
    max_ece_gain: float = 0.05,
) -> tuple[bool, str]:
    d_auc = deployed.get("recent", {}).get("auc", 0.0)
    m_auc = monitored.get("recent", {}).get("auc", 0.0)
    d_ece = deployed.get("recent", {}).get("ece", 0.0)
    m_ece = monitored.get("recent", {}).get("ece", 0.0)

    if m_auc < d_auc - max_regression:
        return True, f"上线后 recent AUC 由 {d_auc:.4f} 跌至 {m_auc:.4f}"
    if m_ece > d_ece + max_ece_gain:
        return True, f"上线后 ECE 由 {d_ece:.4f} 恶化至 {m_ece:.4f}"
    return False, "监控正常"
