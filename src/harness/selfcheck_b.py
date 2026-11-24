# -*- coding: utf-8 -*-
"""用官方训练包真实数据自检 B（轴承 RUL）提交模块。

    python harness/selfcheck_b.py <解压后的训练包目录>

它会：加载官方 checkpoint → 让假 Qwen 选候选 → 续训 → 在公开验证集上算 RMSE
→ 走门禁 → 用 predict() 对一条真实样本出结果。不连平台、不调真实模型。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PKG = Path(sys.argv[1] if len(sys.argv) > 1 else "_pkg").resolve()
sys.path.insert(0, str(PKG))                     # 让 baselines 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import contest_submission_b as sub  # noqa: E402
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402


class FakeParent:
    def __init__(self, pkg: Path, artifact_id: str):
        self.artifact_id = artifact_id
        self.descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": []}
        self._path = pkg / "checkpoint.npz"

    def path(self, rel):
        return self._path


class FakeRound:
    def __init__(self, parent, round_index=1):
        self.parent = parent
        self.round_index = round_index
        self.dataset_key = "B"
        self.max_output_tokens = 200
        self.model_name = "Qwen/Qwen3.5-4B"
        self.task = {"task_type": "regression", "metric": "rmse"}
        self.history_ids = []
        self.call_ids = []
        self.submitted = None

    def history(self):
        return []

    def model(self, messages, **options):
        text = json.dumps({"candidate": 1, "hypothesis": "误差集中在少数轴承，增加续训轮数"}, ensure_ascii=False)
        self.call_ids.append(f"call-{len(self.call_ids) + 1}")

        class R:
            pass

        r = R()
        r.call_id = self.call_ids[-1]
        r.text = text
        r.response = {"choices": [{"message": {"content": text}}]}
        return r

    def submit_candidate(self, directory, **kwargs):
        self.submitted = {"dir": str(directory), **kwargs}
        return {"artifact_id": "art-new"}

    def save_state(self, directory):
        self.state_saved = str(directory)

    def finish(self, **kwargs):
        self.finished = kwargs


def main() -> int:
    os.environ["CONTEST_TRAIN"] = str(PKG / "public" / "train.jsonl")
    os.environ["CONTEST_VALIDATION"] = str(PKG / "public" / "validation.jsonl")
    os.environ["CONTEST_WORK"] = tempfile.mkdtemp(prefix="rsi_b_")

    va = read_dataset(PKG / "public" / "validation.jsonl", "regression")
    base = Model.load(PKG / "checkpoint.npz")
    p0 = base.predict(va["x"])
    print(f"官方 baseline: RMSE={np.sqrt(np.mean((p0 - va['y']) ** 2)):.1f} "
          f"MAE={np.mean(np.abs(p0 - va['y'])):.1f}")

    rnd = FakeRound(FakeParent(PKG, "art-parent"))
    sub.train(rnd)
    print("selected:", rnd.finished["selected_artifact_id"])
    print("reason:", rnd.finished["reason"])

    if rnd.submitted:
        ckpt = Path(rnd.submitted["dir"]) / "checkpoint.npz"
        m = Model.load(ckpt)
        pred = m.predict(va["x"])
        print(f"提交检查点: RMSE={np.sqrt(np.mean((pred - va['y']) ** 2)):.1f} "
              f"MAE={np.mean(np.abs(pred - va['y'])):.1f}")

    sample = type("S", (), {"task": "regression", "inputs": json.loads(
        (PKG / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0])["inputs"]})()
    os.environ["CONTEST_ARTIFACT_PATH"] = str(Path(rnd.submitted["dir"]) / "checkpoint.npz") if rnd.submitted else str(PKG / "checkpoint.npz")
    out = sub.predict(sample, os.path.dirname(os.environ["CONTEST_ARTIFACT_PATH"]))
    print("predict:", out)
    assert out["value"] >= 0
    assert rnd.finished["model_call_ids"], "finish 必须带 model_call_ids"
    print("\n自检通过：续训、门禁、预测三条路径均工作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
