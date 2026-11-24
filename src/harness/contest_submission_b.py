# -*- coding: utf-8 -*-
"""赛题9 · B：滚动轴承剩余寿命预测（regression / RMSE 分钟）

直接复用官方 `baselines` 的 MLP（34→24→1）与 `Model.load/save`，
保证「续训」是真的继续 Adam 状态，而不是另造一个模型。

一轮 train(rnd)：
    1. 读 /data 公开训练/验证集（34 维嵌套特征，按 baselines.data 的路径展开）；
    2. 从 /previous 的 checkpoint.npz 续训（权重 + Adam 一阶/二阶矩 + 标准化/目标缩放）；
    3. 诊断：父模型在验证集上的 RMSE/MAE、以及按轴承的误差分布（最差 2 根）；
    4. 必须调用一次 rnd.model()：让 Qwen 在候选轮数里选一个并给出假设；
    5. 训练 3 个候选（rd 轮数不同），比 RMSE，门禁要求 RMSE 至少降 1% 且 MAE 不变差；
    6. 通过则提交新检查点，否则选回父产物；状态写 /state 供下一轮诊断。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from baselines.data import flatten, read_dataset
from baselines.model import Model

TASKS = ("regression",)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

# 候选：继续训练的轮数（Adam 状态已继承，轮数就是唯一的“预算旋钮”）
GRID = [{"epochs": 2}, {"epochs": 4}, {"epochs": 8}]


def log(*args):
    print(*args, file=sys.stderr)


def rmse_mae(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2))), float(np.mean(np.abs(pred - truth)))


def read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ask_planner(rnd, parent_rmse, parent_mae, worst, grid):
    listing = "\n".join(f"{i}. 继续训练 {c['epochs']} 轮" for i, c in enumerate(grid))
    prompt = (
        "你是滚动轴承剩余寿命预测的训练师。父模型在公开验证集上 "
        f"RMSE={parent_rmse:.1f} 分钟、MAE={parent_mae:.1f} 分钟。\n"
        f"误差最大的轴承：{worst}\n\n候选：\n{listing}\n\n"
        "选一个最可能在有限算力下降低 RMSE、且不会让 MAE 变差的方案，"
        '只输出 JSON：{"candidate": 下标, "hypothesis": "不超过80字"}'
    )
    receipt = rnd.model(
        [{"role": "user", "content": prompt}],
        max_tokens=min(200, rnd.max_output_tokens),
        temperature=0,
    )
    text = receipt.text or ""
    log(f"[qwen] call={receipt.call_id} text={text[:160]!r}")
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        idx = int(obj["candidate"])
        if not 0 <= idx < len(grid):
            raise ValueError("candidate 越界")
        return idx, str(obj["hypothesis"]).strip()
    except Exception as exc:
        log(f"[qwen] 解析失败，改用启发式：{exc}")
        return (1 if parent_rmse > 600 else 0), "按父模型误差水平选择保守的续训轮数"


def load_parent(rnd):
    try:
        for rel in rnd.parent.descriptor.get("checkpoint_paths", []):
            if not str(rel).endswith(".npz"):
                continue
            path = Path(rnd.parent.path(rel))
            if not path.exists():
                continue
            model = Model.load(path)
            log(f"[parent] 载入 {rel}，steps={model.meta.get('steps')} epochs={model.meta.get('epochs')}")
            return model
    except Exception as exc:
        log(f"[parent] 读取失败: {exc}")
    return None


def train(rnd):
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(os.environ.get("CONTEST_TRAIN", "/data/train.jsonl"))
    val_path = Path(os.environ.get("CONTEST_VALIDATION", "/data/validation.jsonl"))
    tr = read_dataset(train_path, "regression")
    va = read_dataset(val_path, "regression")

    parent = load_parent(rnd)
    if parent is None:
        raise RuntimeError("缺少父检查点：本任务必须从官方 baseline 续训")

    pred0 = parent.predict(va["x"])
    parent_rmse, parent_mae = rmse_mae(pred0, va["y"])
    per_bearing = {}
    for i, sid in enumerate(va["ids"]):
        bearing = str(sid).split(":")[0]
        per_bearing.setdefault(bearing, []).append(abs(pred0[i] - va["y"][i]))
    worst = ", ".join(
        f"{b}: {np.mean(v):.0f} 分钟" for b, v in sorted(per_bearing.items(), key=lambda kv: -np.mean(kv[1]))[:2]
    ) or "无"

    idx, hypothesis = ask_planner(rnd, parent_rmse, parent_mae, worst, GRID)
    order = [idx] + [i for i in range(len(GRID)) if i != idx]

    best = None
    for i in order[:3]:
        model = Model.load(rnd.parent.path(
            next(p for p in rnd.parent.descriptor["checkpoint_paths"] if str(p).endswith(".npz"))
        ))
        model.fit(tr["x"], tr["y"], GRID[i]["epochs"])
        rmse, mae = rmse_mae(model.predict(va["x"]), va["y"])
        log(f"[cand {i}] epochs={GRID[i]['epochs']} RMSE={rmse:.1f} MAE={mae:.1f}")
        if best is None or rmse < best["rmse"]:
            best = {"epochs": GRID[i]["epochs"], "model": model, "rmse": rmse, "mae": mae, "index": i}

    gain = (parent_rmse - best["rmse"]) / parent_rmse
    worse_mae = best["mae"] > parent_mae * 1.02
    adopt = gain >= 0.01 and not worse_mae

    if adopt:
        candidate = work / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        best["model"].save(candidate / "checkpoint.npz")
        code = Path(__file__)
        (candidate / code.name).write_text(code.read_text(encoding="utf-8"), encoding="utf-8")
        submitted = rnd.submit_candidate(
            candidate,
            hypothesis=hypothesis,
            change_summary=(
                f"从父检查点续训 {best['epochs']} 轮（Adam 状态继承）；"
                f"验证 RMSE {parent_rmse:.1f} -> {best['rmse']:.1f} 分钟，MAE {parent_mae:.1f} -> {best['mae']:.1f}。"
            ),
            configuration={"epochs": best["epochs"], "candidate_index": best["index"]},
            seed=int(rnd.round_index),
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[code.name],
            predict_argv=PREDICT_ARGV,
        )
        selected = submitted["artifact_id"]
        reason = f"验证 RMSE 降低 {gain * 100:.1f}%（{parent_rmse:.1f}→{best['rmse']:.1f} 分钟），MAE 未变差，通过门禁。"
    else:
        selected = rnd.parent.artifact_id
        reason = (
            f"最优候选 RMSE {best['rmse']:.1f} 相对父模型 {parent_rmse:.1f} 仅改善 "
            f"{gain * 100:.1f}%{ '，且 MAE 变差' if worse_mae else ''}，不满足门禁，保留父模型。"
        )
        log(f"[gate] 回退父产物：{reason}")

    state = {
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "parent_rmse": parent_rmse,
        "parent_mae": parent_mae,
        "best_rmse": float(min(best["rmse"], parent_rmse)),
        "features": len(tr["paths"]),
        "per_bearing_worst": worst,
    }
    (state_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    rnd.save_state(state_dir)

    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis,
        change_summary=f"父模型 RMSE={parent_rmse:.1f}；诊断：误差最大轴承 {worst}",
        reason=reason,
        configuration={"adopted": bool(adopt), "parent_rmse": round(parent_rmse, 1), "best_rmse": round(best["rmse"], 1)},
        seed=int(rnd.round_index),
        history_ids=list(rnd.history_ids),
        model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH") or Path(model_dir) / "checkpoint.npz")
    model = Model.load(path)
    inputs = sample.inputs if hasattr(sample, "inputs") else sample
    paths, values = flatten(inputs)
    mapping = {json.dumps(p, ensure_ascii=False): float(v) for p, v in zip(paths, values)}
    order = [json.dumps(p, ensure_ascii=False) for p in model.meta["paths"]]
    x = np.asarray([[mapping.get(key, 0.0) for key in order]], dtype=np.float64)
    value = float(model.predict(x)[0])
    return {"value": max(0.0, value)}
