"""实验台账：只追加（append-only）的 JSONL，是全系统"可审计"的载体。

每条记录包含：
    exp_id        实验编号（EXP-0001），跨轮次唯一
    parent        父实验编号（形成改进链）
    config        完整超参（可复现）
    config_hash   配置哈希（防篡改比对）
    metrics       双切片指标（stable / recent）
    cost          本轮消耗（算力分钟、标注条数）
    status        base / deploy / reject / rollback
    artifacts     产物路径与其 sha256
    ts            时间戳（UTC）
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with self.path.open(encoding="utf-8") as f:
            self._n = sum(1 for _ in f)

    def next_id(self) -> str:
        self._n += 1
        return f"EXP-{self._n:04d}"

    def append(self, record: dict) -> dict:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> list[dict]:
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def best(self, key: str = "recent") -> dict | None:
        exps = [e for e in self.read_all() if e.get("status") in ("deploy", "base")]
        if not exps:
            return None
        return max(exps, key=lambda e: e["metrics"].get(key, {}).get("auc", 0.0))
