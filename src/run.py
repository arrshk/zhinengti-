"""命令行入口。

用法：
    python run.py --config configs/default.json --outdir runs/demo
    python run.py --config configs/default.json --rounds 1 --quiet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhixun.config import Config
from zhixun.loop import run


def main() -> int:
    ap = argparse.ArgumentParser(description="智训引擎 RSI 闭环")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    out_dir = args.outdir or cfg.output_dir
    summary = run(cfg, Path(out_dir), rounds=args.rounds, verbose=not args.quiet)
    print(f"\n实验台账: {summary['ledger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
