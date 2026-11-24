"""闭环编排器：漂移监测 -> 诊断归因 -> 实验优化 -> 评估上线 -> 回滚。

这是"智训引擎"的主循环。每一轮的输出都落进台账，形成可审计的改进链。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import data as data_mod
from .budget import Budget
from .config import Config
from .diagnose import attribute, to_hypothesis
from .drift import monitor
from .evaluate import evaluate_slices
from .gate import decide
from .ledger import Ledger, sha256_file
from .model import LogisticModel
from .planner import UCBPlanner, build_space, make_planner


def _slices(ds: dict) -> dict:
    """稳定的时间切分（与漂移点无关），保证各轮比较口径一致。"""
    n = len(ds["y"])
    d = ds["drift_start"]
    return {
        "train_base": slice(0, int(d * 0.67)),
        "stable_eval": slice(int(d * 0.67), d),
        "drift_ref": slice(int(d * 0.67), d),
        "recent_train": slice(d, d + int((n - d) * 0.5)),
        "recent_eval": slice(d + int((n - d) * 0.5), n),
    }


def run(cfg: Config, out_dir: str | Path, rounds: int | None = None, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ds = data_mod.make_dataset(
        cfg.data.n_samples,
        cfg.data.n_features,
        cfg.data.drift_start_ratio,
        cfg.data.drift_strength,
        cfg.data.label_noise,
        cfg.data.seed,
    )
    X, y, fnames = ds["X"], ds["y"], ds["feature_names"]
    sl = _slices(ds)

    ledger = Ledger(out_dir / "ledger.jsonl")
    budget = Budget(
        cfg.budget.compute_minutes,
        cfg.budget.annotation_items,
        cfg.budget.minutes_per_train,
        cfg.budget.items_per_label_query,
    )
    rng = np.random.default_rng(cfg.data.seed)

    # ---------- 0. 基线（漂移前的线上模型） ----------
    base = LogisticModel(seed=cfg.data.seed).fit(X[sl["train_base"]], y[sl["train_base"]])
    base_metrics = evaluate_slices(
        base, X, y, {"stable": sl["stable_eval"], "recent": sl["recent_eval"]}
    )
    base_id = ledger.next_id()
    ledger.append(
        {
            "exp_id": base_id,
            "parent": None,
            "round": 0,
            "config": {"name": "baseline", "seed": cfg.data.seed},
            "config_hash": cfg.hash(),
            "metrics": base_metrics,
            "cost": {"minutes": 0.0, "items": 0},
            "status": "base",
            "reason": "漂移前训练的线上基线",
            "artifacts": {},
        }
    )
    if verbose:
        print(f"[base] {base_id} stable={base_metrics['stable']['auc']:.4f} "
              f"recent={base_metrics['recent']['auc']:.4f}")

    deployed = base
    deployed_metrics = base_metrics
    deployed_id = base_id
    history = ledger.read_all()
    planner_kind = os.getenv("ZX_PLANNER", "ucb")
    planner = make_planner(planner_kind, history)

    max_rounds = rounds if rounds is not None else cfg.gate.max_rounds
    events = []

    for r in range(1, max_rounds + 1):
        # ---------- 1. 漂移监测 ----------
        mon = monitor(
            X[sl["drift_ref"]],
            X[sl["recent_train"]],
            fnames,
            psi_threshold=cfg.drift.psi_threshold,
        )
        if verbose:
            print(f"[round {r}] 漂移等级={mon['level']} maxPSI={mon['max_psi']:.3f} "
                  f"top={mon['top_k']}")
        if mon["level"] == "OK" and r > 1:
            events.append({"round": r, "event": "stop", "reason": "无显著漂移"})
            break

        # ---------- 2. 诊断归因 ----------
        w = deployed.w if deployed.w is not None else None
        diag = attribute(
            X[sl["drift_ref"]], X[sl["recent_train"]], fnames,
            weights=w, model_mu=deployed.mu, model_sigma=deployed.sigma, top_k=2,
        )
        hyp = to_hypothesis(diag)
        if verbose:
            print(f"[round {r}] 假设：{hyp['hypothesis']}")

        # ---------- 3. 实验优化（预算感知） ----------
        candidates = build_space(cfg.search_space, hyp["features"], fnames)
        history = ledger.read_all()
        picks = planner.select(
            candidates, cfg.gate.max_candidates_per_round, diagnosis=diag, history=history
        )
        results = []

        for i, cand in enumerate(picks, start=1):
            if not budget.can_afford(items=cfg.budget.items_per_label_query):
                events.append({"round": r, "event": "budget_stop"})
                break

            model = LogisticModel(
                lr=cand["lr"], l2=cand["l2"], epochs=cand["epochs"],
                seed=cfg.data.seed + r, feature_mask=cand["feature_mask"],
                recent_weight=cand["recent_weight"],
            )
            # 训练集 = 历史 + 最新工况，靠 sample weight 让新数据说话
            idx = np.concatenate([
                np.arange(sl["train_base"].start, sl["train_base"].stop),
                np.arange(sl["recent_train"].start, sl["recent_train"].stop),
            ])
            model.fit(X[idx], y[idx])
            budget.spend(f"round{r}-cand{i}", items=cfg.budget.items_per_label_query)
            metrics = evaluate_slices(
                model, X, y, {"stable": sl["stable_eval"], "recent": sl["recent_eval"]}
            )
            decision, reason = decide(
                deployed_metrics, metrics,
                cfg.gate.min_auc_gain, cfg.gate.max_regression,
            )
            exp_id = ledger.next_id()
            ledger.append(
                {
                    "exp_id": exp_id,
                    "parent": deployed_id,
                    "round": r,
                    "hypothesis": hyp["hypothesis"],
                    "config": cand,
                    "config_hash": cfg.hash(),
                    "metrics": metrics,
                    "cost": {"minutes": budget.minutes_per_train,
                             "items": cfg.budget.items_per_label_query},
                    "status": "candidate",
                    "gate": decision,
                    "reason": reason,
                    "artifacts": {},
                }
            )
            results.append((exp_id, cand, metrics, decision, reason, model))
            if verbose:
                print(f"  - {exp_id} rw={cand['recent_weight']} "
                      f"drop={cand['drop_feature']} -> recent={metrics['recent']['auc']:.4f} "
                      f"gate={decision}")

        if not results:
            events.append({"round": r, "event": "no_candidate"})
            break

        # ---------- 4. 评估上线 ----------
        best = max(results, key=lambda t: t[2]["recent"].get("auc", 0.0))
        exp_id, cand, metrics, decision, reason, model = best

        # 同一轮只有一个"最优者"可以进入门禁，其余记为未选中（保证审计链唯一）
        status_map = {t[0]: "reject" for t in results}
        artifacts: dict = {}
        if decision == "deploy":
            deployed, deployed_metrics, deployed_id = model, metrics, exp_id
            model_path = out_dir / f"model_{exp_id}.npz"
            model.save(model_path)
            status_map[exp_id] = "deploy"
            artifacts = {
                "model": model_path.name,
                "sha256": sha256_file(model_path),
            }
            events.append({"round": r, "event": "deploy", "exp": exp_id, "reason": reason})
            if verbose:
                print(f"[round {r}] 上线 {exp_id}（{reason}）")
        else:
            events.append({"round": r, "event": "rollback", "exp": exp_id, "reason": reason})
            if verbose:
                print(f"[round {r}] 回滚：{reason}；继续使用 {deployed_id}")

        # 回填状态与产物哈希：产物与实验一一对应，构成可审计证据
        recs = ledger.read_all()
        for rec in recs:
            if rec["exp_id"] in status_map and rec.get("round") == r:
                rec["status"] = status_map[rec["exp_id"]]
                rec["selected"] = rec["exp_id"] == exp_id
            if rec["exp_id"] == exp_id:
                rec["artifacts"] = artifacts
        (out_dir / "ledger.jsonl").write_text(
            "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs),
            encoding="utf-8",
        )

        if planner_kind == "ucb":
            planner = make_planner(planner_kind, ledger.read_all())

    summary = {
        "rounds_run": len([e for e in events if e["event"] in ("deploy", "rollback")]),
        "deployed": deployed_id,
        "final_metrics": deployed_metrics,
        "budget": budget.summary(),
        "events": events,
        "planner": (
            {"kind": planner_kind, "n_configs": len(getattr(planner, "stats", {}))}
            if planner_kind == "ucb"
            else {"kind": planner_kind, **getattr(planner, "stats", {})}
        ),
        "ledger": str(out_dir / "ledger.jsonl"),
        "n_experiments": len(ledger.read_all()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
