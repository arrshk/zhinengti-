# 智训引擎 · 滚动轴承剩余寿命预测（RSI Harness）

第二届北京市大学生“人工智能+”创新大赛 · 赛题9「智训引擎」
任务 B：轴承剩余寿命预测（regression，RMSE/MAE，单位分钟）

> 一句话：在官方 CPU 基线上做一个**能自己判断该不该继续训**的闭环——
> 诊断 → 让微调后的 Qwen 选训练预算 → 训练候选 → 双门禁 → 上线或回滚。

## 结果（官方公开验证集，1,679 条）

| 版本 | 续训轮数 | RMSE（分钟） | MAE（分钟） | 结论 |
|---|---|---|---|---|
| 官方 baseline | — | 681.2 | 563.1 | 起点 |
| C1 | 2 | 656.4 | 545.9 | 改善不足 |
| **C2** | **4** | **647.2** | **544.1** | **上线（−5.0%）** |
| C3 | 8 | 678.6 | 578.4 | 过拟合，门禁拒绝 |

训练轮数与收益**不单调**：4 轮最优，8 轮 MAE 反而变差 2.7%。这就是“预算约束下自主判断”的意义。

## 快速开始

```bash
pip install -r baselines/requirements.txt     # 仅 numpy
python -m baselines.train --profile B \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 4 --output round-1
```

Harness 提交模块（平台入口）：

```bash
cd src
python harness/selfcheck_b.py ../training_pkg   # 官方数据离线自检
docker build -f harness/Dockerfile.b -t zhixun-bearing-rul:0.1 ..   # 需 Docker
docker run --rm zhixun-bearing-rul:0.1 describe
```

## RSI 闭环怎么实现

```
父检查点 + 公开数据
      │
      ├─ 诊断：父模型 RMSE/MAE + 误差最大的两根轴承
      ├─ 假设：Qwen3.5-4B（LoRA SFT）在编号候选里选一个，给一句话假设
      ├─ 训练：3 个候选全部从父检查点热启动（继承权重 + Adam 矩 + 标准化）
      ├─ 门禁：RMSE 增益 ≥1% 且 MAE 不劣化 >2%
      └─ 选择：通过则提交新检查点，否则选回父产物（回滚，仍写入审计）
```

关键设计：
- **训练师只做选择题**：候选有限、编号、可校验；越界或坏 JSON 自动退回启发式；
- **不许重造模型**：`Model.load` 整体继承 checkpoint，结构/标准化/目标缩放全部沿用；
- **失败也留痕**：`hypothesis / change_summary / reason / model_call_ids` 来自本轮真实动作。

## 目录

```
src/harness/
  contest_submission_b.py   平台提交模块（train / predict）
  selfcheck_b.py            官方数据离线自检
  Dockerfile.b              Harness 镜像
training_pkg/               官方训练包（baseline 代码、checkpoint、公开数据）
src/zhixun/                 闭环演示（漂移监测/归因/UCB/Qwen 训练师/台账/门禁）
src/trainer/                PyTRIO LoRA SFT 脚本
```

## 训练师微调（PyTRIO）

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen3.5-4B` |
| 方法 | LoRA SFT，rank 32，prompt loss 权重 0 |
| 数据 | 120 条「诊断 + 候选 → 选择 + 理由」 |
| loss | 2.0202 → 1.2018（2 epoch，25 秒） |
| 闭环表现 | 2 次调用 / 0 次回退 / 延迟 0.8–1.2 秒 |

## 复现约定

- 训练只用 `public/train.jsonl`；验证集只用于评测与选择，不参与梯度；
- 每轮 seed 固定为轮次号，候选顺序由训练师给定；
- 所有实验记录见 `运行记录.md`，含原始命令与输出。

## 许可

代码用于参赛与复现；官方 `baselines/` 与数据版权归赛事方所有，不在本仓库再分发许可范围内。
