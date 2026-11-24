# 智训引擎 · RSI 闭环参考实现

赛题9《“智训引擎”—工业制造场景下的模型训练与超参数优化》的可运行骨架。
实现 **漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚** 五步闭环，
所有实验写入只追加的 JSONL 台账，满足"可复现、可审计、安全可控"。

## 快速开始

```bash
pip install -r requirements.txt
python run.py --config configs/default.json            # 跑完整闭环
python tests/test_smoke.py                             # 冒烟测试
```

输出目录 `runs/default/`：

| 文件 | 说明 |
|---|---|
| `ledger.jsonl` | 实验台账，每行一个实验，可直接作为"查询依据" |
| `summary.json` | 本次运行的汇总（部署版本、预算、事件） |
| `model_EXP-xxxx.npz` | 上线模型的权重与标准化参数 |
| `config.json` | 本次运行的完整配置（含哈希） |

## 目录

```
zhixun/
  data.py       合成工业数据 + 可控漂移注入
  model.py      轻量逻辑回归（可替换为任意模型，接口不变）
  drift.py      PSI + KS 漂移监测
  diagnose.py   归因 + 工艺知识表 -> 实验假设
  planner.py    UCB 训练师策略 + 搜索空间 + make_planner 工厂
  qwen_planner.py  Qwen 训练师（PyTRIO / OpenAI 兼容），异常自动回退 UCB
  evaluate.py   双切片评估（stable / recent）、AUC/Acc/ECE
  gate.py       上线门禁与回滚判定
  ledger.py     只追加台账 + 产物 sha256
  budget.py     算力/标注预算硬闸
  loop.py       五步闭环编排
trainer/
  build_dataset.py  生成训练师 SFT 数据（chat JSONL）
  train_sft.py      PyTRIO LoRA SFT（含 SwanLab 记录、权重保存）
```

## 两个可切换的“训练师”

```bash
# 默认：UCB，离线可跑，用于本地复现与答辩兜底
python run.py --config configs/default.json

# 切换：Qwen（需先 pip install -U pytrio && trio login，并已训练 adapter）
# 注：ZX_PLANNER=qwen 时，若模型/网络不可用会自动回退 UCB，闭环不会中断
ZX_PLANNER=qwen python run.py --config configs/default.json
```

训练师本体的训练见 `trainer/README.md`（PyTRIO 的 LoRA SFT，基座 Qwen3.5-4B / Qwen3.8-27B）。


## 怎么替换成企业数据

1. 实现 `data.load_enterprise(path)`，返回同样的 dict（`X, y, drift_start, feature_names`）；
2. 若模型需替换，保持 `fit / predict_proba / save / load` 四个方法；
3. 在 `configs/*.json` 中改 `search_space` 与 `budget`，其余不动。

## 二次开发提示

- 想让"训练师"变成 LLM：替换 `planner.UCBPlanner.select()`，输入仍是候选配置与历史台账；
- 想接 SwanLab：在 `loop.py` 每次 `ledger.append` 后调用 `swanlab.log({...})`，
  并把 run 的 URL 写进 `artifacts`，即可满足"云端实验记录链接"要求。
