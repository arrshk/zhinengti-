# Harness 提交模块（轴承健康识别）

平台按轮驱动训练，选手只需要实现 `train(rnd)` 和 `predict(sample, model_dir)`。
RSI 的“自改进”就发生在每一轮里。

## 一轮里发生了什么

```
父检查点 /previous ──┐
上一轮状态 /state  ──┼─> 诊断（Macro-F1、各类召回、特征均值漂移）
公开数据 /data     ──┘        │
                              v
                    rnd.model() 让 Qwen 选候选 + 给假设   ← 平台托管，每轮必须调用
                              │
                              v
              3 个候选（全部从父检查点热启动）训练 + 公开验证集评估
                              │
                              v
                    门禁：Macro-F1 增益 > 0.003 且最差类召回不塌 > 0.02
                        │ 通过                    │ 不通过
                        v                          v
              submit_candidate(新检查点)      选回父产物（回滚）
                        └────────── finish(假设/修改/理由/调用号) ──────────┘
```

## 三个关键设计

1. **续训而不是重造**：`SoftmaxRegression.fit(..., warm=True)` 直接继承父模型的
   `W/b/mean/std`，只在新数据上继续更新——对应赛题“从父检查点续训”的硬要求。
2. **双门禁**：只看 Macro-F1 会掩盖“某个类彻底不认了”。所以再加一条
   “最差类召回不得比父模型低 2pt”，两个条件同时满足才上线。
3. **可审计**：`hypothesis` / `change_summary` / `reason` / `model_call_ids` 全部来自
   本轮真实动作（调用号来自 `rnd.model()` 回执），不是事后编的。

## 本地自检（不连平台）

```bash
cd src
python harness/selfcheck.py
```

会跑三轮假宿主：

| 轮次 | 场景 | 结果 |
|---|---|---|
| R1 | 无漂移、无父模型 | 上线新候选，Macro-F1 0.86 |
| R2 | 工况漂移，父模型掉到 0.15 | 续训后 0.92 上线，最差类 0.88（父 0.03） |
| R3 | 候选被故意设弱 | **门禁回退到父产物**（增益 −0.0004） |

## 构建镜像

```bash
# 在项目根目录（有 contest-harness-sdk-1.0.4/ 和 src/ 的那一层）
docker build -f src/harness/Dockerfile -t zhixun-bearing:0.1 .
docker run --rm zhixun-bearing:0.1 describe
# 期望输出一行 JSON：{"protocol": "...", "tasks": ["classification"], ...}
```

## 接真实 baseline 时要改的两处

1. **父检查点格式**：`load_parent()` 目前认 `npz(W,b,mean,std,classes,features)`。
   官方训练包多半是 sklearn/joblib 或 torch 的 `state_dict`，把这里改成对应加载器即可，
   其余流程不动。
2. **特征顺序**：现在按 `sorted(inputs.keys())` 排序并在 state 里固定；如果 baseline
   有固定列顺序（`feature_names.json` 之类），改成读它，保证跨轮、跨预测一致。

## 注意

- `stdout` 只能出协议 JSON，日志一律 `stderr`（代码里 `log()` 已处理）；
- 每轮至少一次真实模型调用，否则 `finish` 会被平台拒绝；
- PyTRIO 密钥不要进镜像，模型访问走 `rnd.model()` / `rnd.transport()`。
