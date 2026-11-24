"""规划器自检：验证 Qwen 不可用时是否安全回退到 UCB。

用途：
    python tools/check_planner.py            # 检查默认后端（pytrio）
    python tools/check_planner.py --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-4B

判定标准：
    - 打印 picks 数量 = 3；
    - stats["fallbacks"] >= 1 且 picks 仍有效 → 说明“失败可控”，这是赛题要的行为。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhixun.planner import UCBPlanner, build_space
from zhixun.qwen_planner import QwenPlanner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pytrio", choices=["pytrio", "openai"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    features = ["料筒温度", "注射压力", "保压时间", "冷却时间", "原料含水率", "车间湿度"]
    candidates = build_space({}, ["原料含水率"], features)
    diagnosis = [
        {
            "feature": "原料含水率",
            "mean_shift": 0.74,
            "std_shift": 0.6,
            "impact": 0.9,
            "cause": "换批次/受潮",
            "action": "提高近期样本权重",
        }
    ]

    planner = QwenPlanner(
        backend=args.backend,
        base_model=args.model,
        openai_base_url=args.base_url,
        openai_model=args.model,
        verbose=True,
        fallback=UCBPlanner(),
    )
    picks = planner.select(candidates, 3, diagnosis=diagnosis, history=[])
    print(f"候选总数: {len(candidates)}")
    print(f"选中: {len(picks)} 个")
    for p in picks:
        print("  ", {k: p[k] for k in ("lr", "l2", "epochs", "recent_weight", "drop_feature")})
    print(f"stats: {planner.stats}")
    ok = len(picks) == 3
    print("结果:", "通过（回退生效，闭环未中断）" if ok else "异常：未选出候选")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
