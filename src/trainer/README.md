# 训练师（Qwen）训练脚本

用 PyTRIO 对基座做 **LoRA SFT**，把 Qwen 训练成“工业模型训练师”：输入漂移诊断与编号候选配置，输出选择结果 JSON。

## 两条命令

```bash
cd src

# 1) 造数据（demo 合成 120 条；有真实台账就加 --from-ledger runs/default/ledger.jsonl）
python trainer/build_dataset.py --out data/trainer_sft.jsonl --n 120

# 2) 先校验格式，不连云端
python trainer/train_sft.py --dry-run --data data/trainer_sft.jsonl

# 3) 真正训练（需 pip install -U pytrio && trio login）
python trainer/train_sft.py \
  --base-model Qwen/Qwen3.5-4B \
  --rank 32 --epochs 3 --lr 1e-4 \
  --swanlab
```

训练结束会打印 `save_weights_for_sampler` 返回的权重路径，并写 `trainer_adapter_meta.json`（基座、rank、数据哈希、耗时）。

## 基座怎么选

| 基座 | 适用 | 说明 |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 主力，先跑通 | LoRA rank 32 起步；adapter 几十 MB |
| `Qwen/Qwen3.8-27B` | 对照实验 | 同一份数据再训一次，比较成功率与预算 |

**以 `trio.ServiceClient().get_supported_models()` 的返回为准**——官方文档与博客里的版本号并不一致，字符串写错会直接报错。

## 关键实现细节（照官方 quickstart）

1. `enable_thinking=False`：Qwen 的 chat template 必须关思考，否则 completion 里会混入思维链；
2. **prompt 权重设为 0**，只对 completion 计算 loss——训练师只需要学会“怎么选”，不需要复述诊断；
3. 末尾补 `eos_token_id`，并让 weights 同步补 1，否则模型学不会停止；
4. `input_tokens = tokens[:-1]`、`target_tokens = tokens[1:]`，这是标准的 next-token 错位；
5. 每轮 `forward_backward` 之后必须 `optim_step`，且两个 future 都要 `.result()` 才算真正提交。

## 训练完做什么

1. 控制台「权重」页下载 **LoRA adapter**（PyTRIO 不支持全量微调，也不给全量权重）；
2. 把 adapter 与基座名一起登记到 `01_模型配置_模型ID.txt`；
3. 用 adapter 跑规划：`QwenPlanner(base_model=..., model_path=...)`，见 `zhixun/qwen_planner.py`；
4. 在 `05_项目说明书` §5 补一组“基座 vs 微调后”的选择准确率对比——这是训练师是否有效的直接证据。

## 注意

- 合成数据（`--demo`）只用于跑通流程，**不能**作为提交的训练集；
- 训练/采样按量计费，先跑 10–20 步确认 loss 下降，再放大 epoch；
- `trainer_adapter_meta.json` 里的 `data_hash` 要写进实验台账，保证“同一数据可复现”。
