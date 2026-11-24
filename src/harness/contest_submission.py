# -*- coding: utf-8 -*-
"""赛题9 · 轴承健康识别（classification / Macro-F1）

这是**在平台 Harness 里真正跑 RSI 闭环**的提交模块，不是演示脚本。

一轮 train(rnd) 做的事：
    1. 读 /data 的公开训练/验证集，按冻结特征顺序向量化；
    2. 从父检查点 /previous 续训（软启动），并读 /state 里的上一轮特征统计；
    3. 诊断：父模型在验证集上的 Macro-F1 / 各类召回 + 本轮特征均值漂移；
    4. 必须调用一次 rnd.model()：让平台托管的 Qwen 在候选配置里挑一个、并给出本轮假设；
    5. 训练 3 个候选（热启动），在公开验证集上比 Macro-F1 与最差类召回；
    6. 门禁：只有“整体变好且没有类坍塌”才提交新候选，否则**回退到父产物**；
    7. 保存状态（特征顺序、类别、统计量、最好成绩），finish() 写清假设/修改/理由。

对外契约（平台检查）：TASKS / train(rnd) / predict(sample, model_dir)。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

TASKS = ("classification",)          # 本模块只支持分类任务
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
DEFAULT_LABELS = ["normal", "inner_race", "outer_race", "ball"]

# 候选配置：lr / l2 / epochs / 是否按类加权 / 近期样本权重
GRID = [
    {"lr": 0.30, "l2": 1e-4, "epochs": 120, "class_weight": True},
    {"lr": 0.15, "l2": 1e-3, "epochs": 200, "class_weight": True},
    {"lr": 0.50, "l2": 1e-3, "epochs": 80, "class_weight": False},
]


# --------------------------------------------------------------------------- 基础工具
def log(*args):
    """日志一律走 stderr：stdout 是协议通道。"""
    print(*args, file=sys.stderr)


def vectorize(rows, features):
    """把 [{inputs:{...}}] 变成 (X, y, ids)；缺失特征补 0，标签转字符串。"""
    X = np.zeros((len(rows), len(features)), dtype=np.float64)
    y, ids = [], []
    for i, row in enumerate(rows):
        inputs = row.get("inputs") or {}
        for j, name in enumerate(features):
            value = inputs.get(name, 0.0)
            try:
                X[i, j] = float(value)
            except (TypeError, ValueError):
                X[i, j] = 0.0
        y.append(str(row.get("answer")))
        ids.append(row.get("id"))
    return X, y, ids


def macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def per_class_recall(y_true, y_pred, labels):
    out = {}
    for label in labels:
        total = sum(1 for t in y_true if t == label)
        hit = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        out[label] = hit / total if total else 0.0
    return out


# --------------------------------------------------------------------------- 模型
class SoftmaxRegression:
    """多分类逻辑回归；支持从父检查点热启动，便于“续训”而不是重造。"""

    def __init__(self, classes, W=None, b=None, mean=None, std=None):
        self.classes = list(classes)
        self.W = W
        self.b = b
        self.mean = mean
        self.std = std

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=True)
        classes = [str(c) for c in data["classes"]]
        return cls(
            classes,
            W=data["W"].astype(np.float64),
            b=data["b"].astype(np.float64),
            mean=data["mean"].astype(np.float64),
            std=data["std"].astype(np.float64) + 1e-9,
        )

    def save(self, path: Path, features):
        np.savez(
            path,
            W=self.W,
            b=self.b,
            mean=self.mean,
            std=self.std,
            classes=np.asarray(self.classes, dtype=object),
            features=np.asarray(features, dtype=object),
        )

    def _probs(self, X):
        z = ((X - self.mean) / self.std) @ self.W + self.b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return [self.classes[i] for i in self._probs(X).argmax(axis=1)]

    def fit(self, X, y, features, cfg, warm=True, seed=0):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        if self.mean is None or self.mean.shape[0] != d:
            self.mean, self.std = X.mean(0), X.std(0) + 1e-9

        present = sorted(set(y))
        if warm and self.W is not None and d == self.W.shape[0]:
            # 父模型已有的类沿用，新增类补 0 权重
            merged = list(self.classes) + [c for c in present if c not in self.classes]
            W = np.zeros((d, len(merged)))
            b = np.zeros(len(merged))
            W[:, : self.W.shape[1]] = self.W
            b[: self.b.shape[0]] = self.b
            self.W, self.b, self.classes = W, b, merged
        else:
            self.classes = present
            self.W, self.b = np.zeros((d, len(present))), np.zeros(len(present))

        index = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((n, len(self.classes)))
        for i, label in enumerate(y):
            Y[i, index[label]] = 1.0

        weight = np.ones(n)
        if cfg.get("class_weight"):
            counts = {c: max(sum(1 for t in y if t == c), 1) for c in self.classes}
            weight = np.array([n / (len(self.classes) * counts[t]) for t in y])

        Xs = (X - self.mean) / self.std
        for _ in range(int(cfg["epochs"])):
            order = rng.permutation(n)
            for start in range(0, n, 256):
                batch = order[start : start + 256]
                p = self._probs_from(Xs[batch], self.W, self.b)
                grad = (p - Y[batch]) * weight[batch, None]
                self.W -= cfg["lr"] * (Xs[batch].T @ grad / len(batch) + cfg["l2"] * self.W)
                self.b -= cfg["lr"] * grad.mean(0)
        return self

    @staticmethod
    def _probs_from(Xs, W, b):
        z = Xs @ W + b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- 父检查点
def load_parent(rnd, features_hint):
    """尝试读父检查点；读不到就返回 None，并如实记录（不伪造续训）。"""
    try:
        descriptor = rnd.parent.descriptor
        candidates = [p for p in descriptor.get("checkpoint_paths", []) if str(p).endswith((".npz", ".npy"))]
        for rel in candidates:
            path = Path(rnd.parent.path(rel))
            if not path.exists():
                continue
            try:
                model = SoftmaxRegression.load(path)
                log(f"[parent] 已加载 {rel}，类别 {model.classes}")
                return model
            except Exception as exc:  # 父包可能是别的格式
                log(f"[parent] {rel} 无法解析为 SoftmaxRegression: {exc}")
    except Exception as exc:
        log(f"[parent] 读取失败: {exc}")
    return None


# --------------------------------------------------------------------------- Qwen 规划
def ask_planner(rnd, diagnosis, parent_f1, grid):
    """必须调用一次模型；返回 (候选下标, 本轮假设)。解析失败则用启发式兜底。"""
    listing = "\n".join(
        f"{i}. lr={c['lr']} l2={c['l2']} epochs={c['epochs']} class_weight={c['class_weight']}"
        for i, c in enumerate(grid)
    )
    prompt = (
        "你是轴承故障诊断的训练师。当前父模型在公开验证集上的 Macro-F1 为 "
        f"{parent_f1:.4f}。\n诊断信息：\n{diagnosis}\n\n候选配置：\n{listing}\n\n"
        "请选一个最可能提升 Macro-F1、且不会让最差类别坍塌的配置，"
        '只输出 JSON：{"candidate": 下标, "hypothesis": "不超过80字的假设"}'
    )
    receipt = rnd.model(
        [{"role": "user", "content": prompt}],
        max_tokens=min(256, rnd.max_output_tokens),
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
        hypothesis = str(obj.get("hypothesis", "")).strip()
        if not hypothesis:
            raise ValueError("hypothesis 为空")
        return idx, hypothesis
    except Exception as exc:
        log(f"[qwen] 解析失败，改用启发式：{exc}")
        # 启发式：父模型 F1 偏低 → 强正则 + 类加权（第 1 档）
        return (1 if parent_f1 < 0.75 else 0), f"父模型 Macro-F1={parent_f1:.4f}，按启发式选择保守配置"


# --------------------------------------------------------------------------- 一轮训练
def train(rnd):
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)

    train_rows = list(rnd.data.training_rows())
    val_rows = list(rnd.data.validation_rows())
    if not train_rows:
        raise RuntimeError("训练集为空")

    # 特征顺序：优先沿用状态里的，保证跨轮一致
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    prev_state = {}
    try:
        if Path("/state").exists():
            for item in Path("/state").glob("*.json"):
                prev_state = json.loads(item.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[state] 读取失败: {exc}")

    features = prev_state.get("features") or sorted(train_rows[0].get("inputs", {}).keys())
    Xtr, ytr, _ = vectorize(train_rows, features)
    Xva, yva, _ = vectorize(val_rows, features) if val_rows else (np.zeros((0, len(features))), [], [])
    labels = sorted(set(ytr) | set(yva))

    parent = load_parent(rnd, features)
    if parent is not None:
        parent_f1 = macro_f1(yva, parent.predict(Xva), labels) if len(yva) else 0.0
        parent_recall = per_class_recall(yva, parent.predict(Xva), labels) if len(yva) else {}
        parent_worst = min(parent_recall.values()) if parent_recall else 0.0
    else:
        parent_f1, parent_recall, parent_worst = 0.0, {}, 0.0

    # 漂移诊断：本轮训练均值 vs 上轮记录
    drift = []
    if prev_state.get("feature_mean") and len(prev_state["feature_mean"]) == len(features):
        old = np.asarray(prev_state["feature_mean"])
        shift = Xtr.mean(0) - old
        scale = np.asarray(prev_state.get("feature_std") or np.ones_like(old)) + 1e-9
        for name, value in sorted(zip(features, shift / scale), key=lambda kv: -abs(kv[1]))[:3]:
            drift.append(f"{name} 均值偏移 {value:+.2f}σ")
    diagnosis = "；".join(drift) if drift else "首轮/无历史状态，无可用漂移基线"
    if parent_recall:
        worst = min(parent_recall, key=parent_recall.get)
        diagnosis += f"；父模型最差类别 {worst} 召回 {parent_recall[worst]:.3f}"

    idx, hypothesis = ask_planner(rnd, diagnosis, parent_f1, GRID)
    order = [idx] + [i for i in range(len(GRID)) if i != idx]

    # 训练候选（全部从父模型热启动）
    best = None
    for i in order[:3]:
        cfg = GRID[i]
        model = SoftmaxRegression(
            parent.classes if parent else labels,
            W=None if parent is None else parent.W.copy(),
            b=None if parent is None else parent.b.copy(),
            mean=None if parent is None else parent.mean.copy(),
            std=None if parent is None else parent.std.copy(),
        )
        model.fit(Xtr, ytr, features, cfg, warm=parent is not None, seed=rnd.round_index * 10 + i)
        pred = model.predict(Xva) if len(yva) else []
        f1 = macro_f1(yva, pred, labels) if len(yva) else 0.0
        recall = per_class_recall(yva, pred, labels) if len(yva) else {}
        worst = min(recall.values()) if recall else 0.0
        log(f"[cand {i}] f1={f1:.4f} worst_recall={worst:.3f} cfg={cfg}")
        if best is None or f1 > best["f1"]:
            best = {"cfg": cfg, "model": model, "f1": f1, "worst": worst, "index": i}

    # 门禁：整体要涨，最差类不能塌
    gain = best["f1"] - parent_f1
    collapsed = best["worst"] < parent_worst - 0.02
    adopt = (gain > 0.003) and not collapsed

    if adopt:
        candidate = work / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        ckpt = candidate / "checkpoint.npz"
        best["model"].save(ckpt, features)
        code = Path(__file__)
        (candidate / code.name).write_text(code.read_text(encoding="utf-8"), encoding="utf-8")
        submitted = rnd.submit_candidate(
            candidate,
            hypothesis=hypothesis,
            change_summary=(
                f"从父检查点热启动，采用 lr={best['cfg']['lr']}、l2={best['cfg']['l2']}、"
                f"epochs={best['cfg']['epochs']}、class_weight={best['cfg']['class_weight']}；"
                f"公开验证 Macro-F1 {parent_f1:.4f} -> {best['f1']:.4f}。"
            ),
            configuration={**best["cfg"], "diagnosis": diagnosis, "candidate_index": best["index"]},
            seed=rnd.round_index * 10 + best["index"],
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[code.name],
            predict_argv=PREDICT_ARGV,
        )
        selected = submitted["artifact_id"]
        reason = f"公开验证 Macro-F1 提升 {gain:+.4f}，最差类召回 {best['worst']:.3f}（父 {parent_worst:.3f}），通过门禁。"
    else:
        selected = rnd.parent.artifact_id
        reason = (
            f"候选最优 Macro-F1 {best['f1']:.4f} 相对父模型 {parent_f1:.4f} 增益 {gain:+.4f}"
            f"{'，且最差类坍塌' if collapsed else ''}，不满足门禁，保留父模型。"
        )
        log(f"[gate] 回退父产物：{reason}")

    state = {
        "features": features,
        "labels": labels,
        "feature_mean": Xtr.mean(0).tolist(),
        "feature_std": (Xtr.std(0) + 1e-9).tolist(),
        "best_f1": float(max(best["f1"], parent_f1)),
        "round_index": rnd.round_index,
        "dataset_key": rnd.dataset_key,
    }
    (state_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    rnd.save_state(state_dir)

    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis,
        change_summary=f"诊断：{diagnosis}",
        reason=reason,
        configuration={"adopted": bool(adopt), "parent_f1": round(parent_f1, 4), "best_f1": round(best["f1"], 4)},
        seed=rnd.round_index,
        history_ids=list(rnd.history_ids),
        model_call_ids=list(rnd.call_ids),
    )


# --------------------------------------------------------------------------- 预测
def predict(sample, model_dir):
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH") or Path(model_dir) / "checkpoint.npz")
    data = np.load(path, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]
    features = [str(f) for f in data["features"]]
    row = sample.inputs if hasattr(sample, "inputs") else sample
    X = np.zeros((1, len(features)))
    for j, name in enumerate(features):
        try:
            X[0, j] = float(row.get(name, 0.0))
        except (TypeError, ValueError):
            X[0, j] = 0.0

    mean, std = data["mean"], data["std"] + 1e-9
    z = ((X - mean) / std) @ data["W"] + data["b"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)[0]
    p = p / p.sum()
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()

    best = int(np.argmax(p))
    out = [{"label": label, "score": float(p[i])} for i, label in enumerate(classes)]
    out.sort(key=lambda item: -item["score"])
    # 保证最高分唯一，避免平台判定"最高分不唯一"
    out[0]["score"] = float(min(1.0, out[0]["score"] + 1e-6))
    out = [{"label": item["label"], "score": item["score"]} for item in out]
    assert best >= 0
    return out
