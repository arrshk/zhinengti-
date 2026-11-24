"""生成训练师 SFT 数据集（chat 格式 JSONL）。

每条样本 = 一条“状态 -> 专家选择”：
    user      : 漂移诊断 + 近期实验 + 编号候选配置
    assistant : {"picks":[...],"reason":"..."}

真实数据来源建议：
    1) 你们跑过的台账 runs/default/ledger.jsonl（每个 round 的“最优候选”就是专家答案）；
    2) 补充人工标注（工程师认为该选哪个），几十条即可起步。

本脚本的 --demo 会合成 N 条同构样本，仅用于把训练流程跑通；**不要**把合成数据当作
真实训练集提交，评测时会被问“你们的 SFT 数据从哪来”。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "你是工业制造场景的模型训练师。给定当前数据漂移诊断与一组编号候选配置，"
    "你要为下一轮训练挑选最可能带来提升的配置。"
    "只能返回 JSON：{\"picks\": [下标...], \"reason\": \"不超过60字\"}。"
)


def make_state(rng: random.Random, n_cand: int = 6) -> tuple[str, str]:
    feature = rng.choice(["原料含水率", "车间湿度", "料筒温度", "注射压力"])
    shift = rng.uniform(0.3, 1.2)
    psi = rng.uniform(0.12, 0.55)
    diag = f"- {feature}: 均值偏移 {shift:+.3f}，PSI {psi:.2f}，可能原因换批次/设备老化"
    rows = []
    for i in range(n_cand):
        rows.append(
            f"{i}. lr={rng.choice([0.05,0.15,0.3])} l2={rng.choice([1e-4,1e-3,1e-2])} "
            f"epochs={rng.choice([80,160])} recent_weight={rng.choice([1.0,3.0,8.0])} "
            f"drop={rng.choice(['None','原料含水率','车间湿度'])}"
        )
    user = (
        f"【漂移诊断】\n{diag}\n\n"
        f"【近期实验】\n- 暂无历史实验\n\n"
        f"【候选配置】\n" + "\n".join(rows) + "\n\n请选出最多 3 个最值得跑的配置下标。"
    )
    # 专家策略（演示用规则）：漂移显著时提高近期权重并丢弃漂移特征
    best = sorted(
        range(n_cand),
        key=lambda i: ("8.0" in rows[i]) + ("原料含水率" in rows[i] or "车间湿度" in rows[i]),
        reverse=True,
    )[:3]
    reason = f"{feature} 漂移明显，优先加大近期权重并丢弃该特征"
    answer = json.dumps({"picks": best, "reason": reason}, ensure_ascii=False)
    return user, answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/trainer_sft.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-ledger", default=None, help="可选：从 ledger.jsonl 抽取真实样本")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = []
    if args.from_ledger:
        recs = [json.loads(l) for l in Path(args.from_ledger).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in recs:
            if r.get("status") != "deploy":
                continue
            user = (
                f"【漂移诊断】\n- {r.get('hypothesis', '漂移')}\n\n"
                f"【近期实验】\n- 暂无\n\n"
                f"【候选配置】\n0. 见台账 {r['exp_id']}\n\n请选出最多 3 个下标。"
            )
            answer = json.dumps({"picks": [0], "reason": r.get("reason", "")[:60]}, ensure_ascii=False)
            samples.append((user, answer))

    while len(samples) < args.n:
        samples.append(make_state(rng))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for user, answer in samples:
            f.write(json.dumps([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ], ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
