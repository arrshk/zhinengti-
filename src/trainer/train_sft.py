"""用 PyTRIO 对「训练师」做 LoRA SFT，产出可下载的 adapter。

流程与 PyTRIO 官方 quickstart 一致：
    1. ServiceClient -> create_lora_training_client(base_model, rank)
    2. 数据集转 Datum（prompt 权重 0，只学 completion，末尾补 EOS）
    3. 循环 forward_backward("cross_entropy") + optim_step(AdamParams)
    4. save_weights_for_sampler() -> 得到权重路径，可立即用于 sampling
    5. 训练完在控制台「权重」页下载 adapter（docs.pytrio.com/guide/download）

用法：
    # 先不连云端，检查数据格式
    python trainer/train_sft.py --dry-run

    # 真正训练（需先 trio login）
    python trainer/train_sft.py --base-model Qwen/Qwen3.5-4B --rank 32 --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "你是工业制造场景的模型训练师。给定当前数据漂移诊断与一组编号候选配置，"
    "你要为下一轮训练挑选最可能带来提升的配置。"
    "只能返回 JSON：{\"picks\": [下标...], \"reason\": \"不超过60字\"}。"
)


def load_jsonl(path: Path) -> list[list[dict]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"数据集为空: {path}")
    return rows


def process_example(messages: list[dict], tokenizer):
    """prompt 部分权重 0，completion 部分权重 1；末尾补 EOS。"""
    import numpy as np
    import pytrio as trio

    prompt_messages = messages[:-1]
    completion = messages[-1]["content"]

    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    eos = tokenizer.eos_token_id
    if eos is not None:
        tokens.append(eos)
        weights.append(1)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": np.asarray(loss_weights, dtype=np.float32),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trainer_sft.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="zhixun-trainer")
    ap.add_argument("--swanlab", action="store_true", help="同时记录到 SwanLab")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据，不连云端")
    args = ap.parse_args()

    data_path = Path(args.data)
    samples = load_jsonl(data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]

    if args.dry_run:
        lens = [len(s) for s in samples]
        print(f"[dry-run] 样本 {len(samples)} 条；消息条数 {min(lens)}–{max(lens)}；"
              f"数据哈希 {digest}")
        bad = [i for i, s in enumerate(samples) if s[-1].get("role") != "assistant"]
        print(f"[dry-run] 末尾非 assistant 的样本：{bad or '无'}")
        print(f"[dry-run] 提示：真正训练需先 `pip install pytrio && trio login`")
        return

    import numpy as np
    import pytrio as trio

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model, rank=args.rank
    )
    tokenizer = training_client.get_tokenizer()

    processed = [process_example(s, tokenizer) for s in samples]
    print(f"数据 {len(processed)} 条，基座 {args.base_model}，LoRA rank {args.rank}")

    run = None
    if args.swanlab:
        try:
            import swanlab

            run = swanlab.init(
                project="zhixun-trainer",
                experiment_name=args.name,
                config={"base_model": args.base_model, "rank": args.rank,
                        "epochs": args.epochs, "lr": args.lr, "data_hash": digest},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] SwanLab 未启用：{exc}")

    t0 = time.time()
    for epoch in range(args.epochs):
        fwdbwd = training_client.forward_backward(processed, "cross_entropy")
        optim = training_client.optim_step(trio.AdamParams(learning_rate=args.lr))
        fwdbwd.result()
        optim.result()

        # 用本次 forward 的 logprobs 与权重算平均 loss
        fw = fwdbwd.result()
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fw.loss_fn_outputs])
        weights = np.concatenate([e.loss_fn_inputs["weights"].tolist() for e in processed])
        loss = float(-np.dot(logprobs, weights) / weights.sum())
        print(f"epoch {epoch + 1}/{args.epochs}  loss/ token = {loss:.4f}")
        if run is not None:
            run.log({"train/loss": loss, "epoch": epoch + 1})

    saved = training_client.save_weights_for_sampler(name=args.name)
    path = saved.result().path
    meta = {
        "base_model": args.base_model,
        "lora_rank": args.rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "data": str(data_path),
        "data_hash": digest,
        "n_samples": len(processed),
        "weights_path": path,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path("trainer_adapter_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n下一步：PyTRIO 控制台 → 权重 → 下载 LoRA adapter；"
          "把 adapter 文件名与基座全名写入 01_模型配置_模型ID.txt")


if __name__ == "__main__":
    main()
