"""训练师（Qwen）规划器：让微调后的 Qwen 从候选配置里做选择。

设计要点（对应赛题“安全可控”）：
    1. LLM **只做选择，不做生成**——给它编号的候选清单，让它返回下标；
       这样从机制上杜绝“编出不存在的超参”；
    2. 返回值必须通过白名单校验（下标在范围内、去重、不超过 k 个）；
    3. 任何异常（超时、JSON 解析失败、越界）都**回退到 UCB**，保证闭环不中断。

两种后端：
    - "pytrio"：用 PyTRIO 的 sampling client（微调后的 adapter 直接推理）
    - "openai"：任何 OpenAI 兼容网关（便于本地/企业内网部署）
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .planner import UCBPlanner

SYSTEM_PROMPT = (
    "你是工业制造场景的模型训练师。给定当前数据漂移诊断与一组编号候选配置，"
    "你要为下一轮训练挑选最可能带来提升的配置。"
    "只能返回 JSON：{\"picks\": [下标...], \"reason\": \"不超过60字\"}。"
    "不要输出任何其他文字。"
)


def _compact(candidates: list[dict]) -> str:
    rows = []
    for i, c in enumerate(candidates):
        rows.append(
            f"{i}. lr={c['lr']} l2={c['l2']} epochs={c['epochs']} "
            f"recent_weight={c['recent_weight']} drop={c.get('drop_feature')}"
        )
    return "\n".join(rows)


class QwenPlanner:
    def __init__(
        self,
        candidates_k: int = 3,
        backend: str = "pytrio",
        base_model: str = "Qwen/Qwen3.5-4B",
        model_path: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        fallback: UCBPlanner | None = None,
        verbose: bool = False,
    ):
        self.k = candidates_k
        self.backend = backend
        self.base_model = base_model
        self.model_path = model_path or os.environ.get("ZX_QWEN_WEIGHTS") or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = fallback or UCBPlanner()
        self.verbose = verbose
        self._sampler = None
        self._tokenizer = None
        self._openai = None
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.stats = {"calls": 0, "fallbacks": 0, "latency_ms": []}

    # ---------------- 后端 ----------------
    def _ensure_backend(self):
        if self.backend == "pytrio":
            if self._sampler is not None:
                return
            import pytrio as trio  # 延迟导入：无账号/无网络时不拖垮主流程

            client = trio.ServiceClient()
            kwargs: dict[str, Any] = {"base_model": self.base_model}
            if self.model_path:
                kwargs["model_path"] = self.model_path
            self._sampler = client.create_sampling_client(**kwargs)
            self._tokenizer = self._sampler.get_tokenizer()
        elif self.backend == "openai":
            if self._openai is not None:
                return
            from openai import OpenAI

            self._openai = OpenAI(base_url=self.openai_base_url, api_key="EMPTY")
        else:
            raise ValueError(f"未知后端: {self.backend}")

    # ---------------- 提示词 ----------------
    def _prompt(self, diagnosis: list[dict], history: list[dict], candidates: list[dict]) -> str:
        diag_txt = "\n".join(
            f"- {d['feature']}: 均值偏移 {d['mean_shift']:+.3f}，可能原因 {d['cause']}"
            for d in diagnosis
        ) or "- 无明显漂移"
        hist_txt = "\n".join(
            f"- {e['exp_id']} {e.get('config', {}).get('recent_weight')} "
            f"recent_auc={e['metrics']['recent']['auc']:.4f} status={e['status']}"
            for e in history[-5:]
        ) or "- 暂无历史实验"
        return (
            f"【漂移诊断】\n{diag_txt}\n\n"
            f"【近期实验】\n{hist_txt}\n\n"
            f"【候选配置】\n{_compact(candidates)}\n\n"
            f"请选出最多 {self.k} 个最值得跑的配置下标。"
        )

    # ---------------- 调用 ----------------
    def _ask(self, prompt: str) -> str:
        if self.backend == "pytrio":
            import pytrio as trio

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            params = trio.SamplingParams(
                max_tokens=self.max_tokens, temperature=self.temperature, seed=0
            )
            fut = self._sampler.sample(
                prompt=trio.ModelInput.from_ints(self._tokenizer.encode(text, add_special_tokens=False)),
                sampling_params=params,
                num_samples=1,
            )
            return fut.result().sequences[0].text

        rsp = self._openai.chat.completions.create(
            model=self.openai_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return rsp.choices[0].message.content

    # ---------------- 主接口（与 UCBPlanner 对齐） ----------------
    def select(self, candidates: list[dict], k: int | None = None,
               diagnosis: list[dict] | None = None,
               history: list[dict] | None = None) -> list[dict]:
        if k:
            self.k = k
        try:
            self._ensure_backend()
            t0 = time.time()
            raw = self._ask(self._prompt(diagnosis or [], history or [], candidates))
            self.stats["latency_ms"].append(int((time.time() - t0) * 1000))
            self.stats["calls"] += 1

            picks = self._parse(raw)
            chosen = [candidates[i] for i in picks if 0 <= i < len(candidates)]
            # 去重后不足一个则回退
            uniq, seen = [], set()
            for c in chosen:
                sig = (c.get("drop_feature"), c["recent_weight"], c["epochs"])
                if sig not in seen:
                    seen.add(sig)
                    uniq.append(c)
            if not uniq:
                raise ValueError(f"LLM 未给出有效下标: {raw[:120]}")
            return uniq[: self.k]
        except Exception as exc:  # noqa: BLE001 - 这里必须兜底
            self.stats["fallbacks"] += 1
            if self.verbose:
                print(f"[qwen_planner] 回退 UCB：{type(exc).__name__}: {exc}")
            return self.fallback.select(candidates, self.k)

    @staticmethod
    def _parse(raw: str) -> list[int]:
        raw = raw.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("返回不含 JSON")
        obj = json.loads(raw[start : end + 1])
        picks = obj.get("picks", [])
        if not isinstance(picks, list):
            raise ValueError("picks 不是数组")
        return [int(p) for p in picks][:5]
