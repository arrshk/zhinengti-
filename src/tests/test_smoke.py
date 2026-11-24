"""冒烟测试：不依赖 pytest，直接 python tests/test_smoke.py 即可运行。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from zhixun import data as data_mod
from zhixun.budget import Budget
from zhixun.config import Config
from zhixun.drift import ks_statistic, monitor, psi
from zhixun.evaluate import auc
from zhixun.loop import run


class TestPrimitives(unittest.TestCase):
    def test_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc(y, s), 1.0, places=6)

    def test_auc_inverted(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(auc(y, s), 0.0, places=6)

    def test_psi_same_distribution(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)
        self.assertLess(psi(a, a.copy()), 1e-6)

    def test_ks_detects_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(2, 1, 1000)
        self.assertGreater(ks_statistic(a, b), 0.5)


class TestPipeline(unittest.TestCase):
    def test_drift_detected(self):
        ds = data_mod.make_dataset(2000, 6, 0.6, 1.0, 0.05, 7)
        d = ds["drift_start"]
        mon = monitor(ds["X"][:d], ds["X"][d:], ds["feature_names"])
        self.assertIn(mon["level"], ("WARN", "SEVERE"))

    def test_budget_guard(self):
        b = Budget(compute_minutes=5, annotation_items=100, minutes_per_train=2.5)
        b.spend("a")
        b.spend("b")
        self.assertFalse(b.can_afford())
        with self.assertRaises(RuntimeError):
            b.spend("c")

    def test_loop_writes_ledger(self):
        cfg = Config.load(Path(__file__).resolve().parents[1] / "configs" / "default.json")
        cfg.data.n_samples = 1500
        cfg.gate.max_rounds = 1
        cfg.gate.max_candidates_per_round = 2
        tmp = Path(__file__).resolve().parents[1] / "runs" / "_smoke"
        s = run(cfg, tmp, rounds=1, verbose=False)
        self.assertTrue(Path(s["ledger"]).exists())
        self.assertGreaterEqual(s["n_experiments"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
