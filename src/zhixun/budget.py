"""预算账本：算力分钟与标注条数的硬约束。

赛题原文要求"在算力与标注预算约束下"自主决策，因此预算不是事后统计，
而是**决策前置条件**：买不起的实验直接不生成，而不是跑完再解释。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Budget:
    compute_minutes: float
    annotation_items: int
    minutes_per_train: float = 2.5
    items_per_label_query: int = 50

    spent_minutes: float = 0.0
    spent_items: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def minutes_left(self) -> float:
        return max(self.compute_minutes - self.spent_minutes, 0.0)

    @property
    def items_left(self) -> int:
        return max(self.annotation_items - self.spent_items, 0)

    def can_afford(self, minutes: float | None = None, items: int = 0) -> bool:
        m = self.minutes_per_train if minutes is None else minutes
        return self.minutes_left >= m and self.items_left >= items

    def spend(self, reason: str, minutes: float | None = None, items: int = 0) -> None:
        m = self.minutes_per_train if minutes is None else minutes
        if not self.can_afford(m, items):
            raise RuntimeError(f"预算不足：需要 {m} 分钟 / {items} 条，"
                               f"剩余 {self.minutes_left:.1f} 分钟 / {self.items_left} 条")
        self.spent_minutes += m
        self.spent_items += items
        self.log.append(
            {"reason": reason, "minutes": m, "items": items,
             "minutes_left": self.minutes_left, "items_left": self.items_left}
        )

    def summary(self) -> dict:
        return {
            "compute_minutes": self.compute_minutes,
            "spent_minutes": round(self.spent_minutes, 2),
            "minutes_left": round(self.minutes_left, 2),
            "annotation_items": self.annotation_items,
            "spent_items": self.spent_items,
            "items_left": self.items_left,
            "utilization": round(self.spent_minutes / max(self.compute_minutes, 1e-9), 3),
        }
