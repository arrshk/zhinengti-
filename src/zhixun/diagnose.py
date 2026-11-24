"""诊断归因：从"哪个特征漂了"进一步回答"为什么会漂、该怎么办"。

工业场景里，评委最关心的不是 PSI 数值本身，而是能否从漂移走到可执行的
实验假设。这里用三步法：
    1) 分布维度：PSI/KS 排序（见 drift 模块）；
    2) 贡献维度：用当前线上模型的权重 × 特征偏移量，估计对预测的冲击；
    3) 知识维度：特征 -> 可能原因 -> 建议动作（工艺知识表）。
"""

from __future__ import annotations

import numpy as np

# 工艺知识表：把统计信号翻译成工程假设。真实比赛应替换为企业工艺文档中的表述。
KNOWLEDGE_BASE = {
    "原料含水率": {
        "cause": "换批次/受潮导致原料含水率上升，塑化过程产生气泡",
        "action": "在训练窗口中对近期样本加权，并加入 含水率×湿度 交互特征",
        "owner": "工艺工程师 + 数据",
    },
    "车间湿度": {
        "cause": "季节变化或除湿设备效率下降",
        "action": "加入环境特征分箱，按季度分层评估",
        "owner": "设备工程师",
    },
    "料筒温度": {
        "cause": "加热圈老化或温控 PID 参数漂移",
        "action": "检查设备维护记录；对温度做分段归一化",
        "owner": "设备工程师",
    },
    "注射压力": {
        "cause": "液压系统压力波动或模具磨损",
        "action": "加入压力-时间曲线特征；缩短重训周期",
        "owner": "设备工程师",
    },
    "保压时间": {
        "cause": "工艺参数被人工调整（换模试产）",
        "action": "核对 MES 工艺参数变更记录",
        "owner": "工艺工程师",
    },
    "冷却时间": {
        "cause": "环境温度变化导致冷却效率变化",
        "action": "引入环境温度交互项",
        "owner": "工艺工程师",
    },
}


def attribute(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    weights: np.ndarray | None = None,
    model_mu: np.ndarray | None = None,
    model_sigma: np.ndarray | None = None,
    top_k: int = 2,
) -> list[dict]:
    """对 Top-K 漂移特征给出"分布偏移 + 预测冲击 + 工程假设"。"""
    out = []
    for j, name in enumerate(feature_names):
        shift = float(np.mean(current[:, j]) - np.mean(reference[:, j]))
        if model_sigma is not None and model_sigma[j] > 0:
            shift_std = shift / float(model_sigma[j])
        else:
            shift_std = shift
        impact = abs(float(weights[j]) * shift_std) if weights is not None else 0.0
        kb = KNOWLEDGE_BASE.get(name, {})
        out.append(
            {
                "feature": name,
                "mean_shift": shift,
                "std_shift": shift_std,
                "impact": impact,
                "cause": kb.get("cause", "待补充"),
                "action": kb.get("action", "待补充"),
                "owner": kb.get("owner", "待补充"),
            }
        )

    out.sort(key=lambda r: r["impact"], reverse=True)
    return out[:top_k]


def to_hypothesis(diagnosis: list[dict]) -> dict:
    """把归因结果收敛成一条可执行假设（供实验规划器使用）。"""
    if not diagnosis:
        return {"hypothesis": "无明显漂移", "actions": []}
    top = diagnosis[0]
    return {
        "hypothesis": f"{top['feature']} 发生漂移（均值偏移 {top['mean_shift']:+.3f}），"
        f"推测原因：{top['cause']}",
        "actions": [d["action"] for d in diagnosis],
        "features": [d["feature"] for d in diagnosis],
    }
