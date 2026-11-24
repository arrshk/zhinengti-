# -*- coding: utf-8 -*-
"""离线自检：用假宿主跑两轮 RSI，验证「续训 -> 诊断 -> Qwen 选候选 -> 门禁 -> 回退」链路。

不连接平台、不调用真实模型，纯本地 numpy。
    python harness/selfcheck.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contest_submission as sub  # noqa: E402

LABELS = ["normal", "inner_race", "outer_race", "ball"]
FEATURES = [f"f{i}" for i in range(8)]


def make_round_data(rng, n, drift=0.0, noise=0.05):
    """合成轴承特征：每个类一个独立方向的簇。

    drift>0 时同时做两件事：逐特征缩放（工况变了）+ 类间距压缩（可分性变差），
    这样父模型会真的退化，续训才有意义。
    """
    per = n // len(LABELS)
    dim = len(FEATURES)
    centers = rng.normal(0, 1.0, (len(LABELS), dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
    centers *= 3.0

    scale = np.ones(dim)
    if drift:
        scale[::2] = 1.0 + 0.5 * drift          # 一半特征被放大
        centers = centers @ np.diag(scale)      # 类中心跟着变形
        centers *= 1.0 / (1.0 + 0.35 * drift)   # 类间距被压缩

    X, y = [], []
    for k, label in enumerate(LABELS):
        X.append(rng.normal(centers[k], 0.9, (per, dim)) * scale)
        y += [label] * per
    X = np.vstack(X)
    idx = rng.permutation(len(y))
    X, y = X[idx], [y[i] for i in idx]
    for i in range(len(y)):
        if rng.random() < noise:
            y[i] = LABELS[rng.integers(0, len(LABELS))]
    rows = [{"id": f"s{i}", "inputs": dict(zip(FEATURES, map(float, X[i]))), "answer": y[i]}
            for i in range(len(y))]
    return rows


class FakeParent:
    def __init__(self, root: Path, artifact_id: str, ckpt: str):
        self.root = root
        self.artifact_id = artifact_id
        self.descriptor = {"checkpoint_paths": [ckpt], "code_paths": []}

    def path(self, rel):
        return self.root / rel


class FakeRound:
    def __init__(self, work: Path, train_rows, val_rows, parent, round_index=1):
        self.work = work
        self.round_index = round_index
        self.dataset_key = "A"
        self.max_output_tokens = 256
        self.model_name = "Qwen/Qwen3.5-4B"
        self.task = {"task_type": "classification", "metric": "macro_f1"}
        self.parent = parent
        self.history_ids = []
        self.call_ids = []
        self._train, self._val = train_rows, val_rows
        self.submitted = None

        outer = self

        class Data:
            def training_rows(self_inner):
                return iter(outer._train)

            def validation_rows(self_inner):
                return iter(outer._val)

        self.data = Data()

    def history(self):
        return []

    def model(self, messages, **options):
        # 假模型：总选第 0 个候选，并给出结构化假设
        text = json.dumps({"candidate": 0, "hypothesis": "工况有漂移，提高学习率并保持类加权"}, ensure_ascii=False)
        self.call_ids.append(f"call-{len(self.call_ids) + 1}")

        class Receipt:
            pass

        r = Receipt()
        r.call_id = self.call_ids[-1]
        r.response = {"choices": [{"message": {"content": text}}]}
        r.text = text
        return r

    def submit_candidate(self, directory, **kwargs):
        self.submitted = {"dir": str(directory), **kwargs}
        return {"artifact_id": "art-new", "validation": {"macro_f1": 0.9}}

    def save_state(self, directory):
        self.state_saved = str(directory)

    def finish(self, **kwargs):
        self.finished = kwargs


def main() -> int:
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(prefix="rsi_"))
    os.environ["CONTEST_WORK"] = str(tmp / "ws")

    # ---- 第 1 轮：无漂移，父检查点不存在（平台首轮给的是官方 baseline） ----
    rows1 = make_round_data(rng, 1600, drift=0.0)
    r1 = FakeRound(
        tmp,
        rows1[:1200],
        rows1[1200:],
        FakeParent(tmp / "prev1", "art-parent", "checkpoint.npz"),
        round_index=1,
    )
    (tmp / "prev1").mkdir(parents=True, exist_ok=True)
    sub.train(r1)
    print("R1 adopted:", r1.finished["selected_artifact_id"], "| reason:", r1.finished["reason"])

    # 把第 1 轮的候选当作第 2 轮的父检查点
    prev2 = tmp / "prev2"
    prev2.mkdir(parents=True, exist_ok=True)
    ckpt_src = Path(r1.submitted["dir"]) / "checkpoint.npz"
    (prev2 / "checkpoint.npz").write_bytes(ckpt_src.read_bytes())

    # ---- 第 2 轮：发生漂移，父模型应被续训/回退 ----
    rows2 = make_round_data(rng, 1600, drift=1.0)
    r2 = FakeRound(
        tmp,
        rows2[:1200],
        rows2[1200:],
        FakeParent(prev2, "art-new", "checkpoint.npz"),
        round_index=2,
    )
    sub.train(r2)
    print("R2 selected:", r2.finished["selected_artifact_id"], "| reason:", r2.finished["reason"])

    # ---- 第 3 轮：同分布，但候选被故意设弱，验证门禁会回退到父产物 ----
    prev3 = tmp / "prev3"
    prev3.mkdir(parents=True, exist_ok=True)
    ckpt2 = Path(r2.submitted["dir"]) / "checkpoint.npz"
    (prev3 / "checkpoint.npz").write_bytes(ckpt2.read_bytes())

    rows3 = make_round_data(rng, 1600, drift=1.0)
    r3 = FakeRound(
        tmp,
        rows3[:1200],
        rows3[1200:],
        FakeParent(prev3, "art-R2", "checkpoint.npz"),
        round_index=3,
    )
    sub.GRID = [{"lr": 1e-4, "l2": 10.0, "epochs": 1, "class_weight": True}] * 3
    sub.train(r3)
    print("R3 selected:", r3.finished["selected_artifact_id"], "| reason:", r3.finished["reason"])
    assert r3.finished["selected_artifact_id"] == "art-R2", "门禁未回退到父产物"

    # ---- 预测接口 ----
    sample = type("S", (), {"task": "classification", "inputs": {f: 1.0 for f in FEATURES}})()
    model_dir = Path(r1.submitted["dir"])
    os.environ["CONTEST_ARTIFACT_PATH"] = str(model_dir / "checkpoint.npz")
    out = sub.predict(sample, str(model_dir))
    total = sum(item["score"] for item in out)
    print("predict:", out, "sum=", round(total, 4))
    assert abs(total - 1.0) < 1e-6, "概率未归一"
    assert len({item["label"] for item in out}) == len(out), "标签重复"
    assert r2.finished["model_call_ids"], "finish 必须带 model_call_ids"

    print("\n自检通过：两轮闭环、门禁与预测均按契约工作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
