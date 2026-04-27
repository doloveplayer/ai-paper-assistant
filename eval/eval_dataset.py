"""评估数据集 —— dataclass 模型 + JSON 加载器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_CATEGORIES = {"methodology", "dataset", "results", "architecture", "comparison", "general"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class EvalCase:
    """单个评估测试用例。"""

    id: str
    query: str
    paper_id: str | None = None
    paper_name: str | None = None
    gt_relevant_pages: list[int] = field(default_factory=list)
    gt_answer_facts: list[str] = field(default_factory=list)
    gt_keywords: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        cat = d.get("category", "general")
        if cat not in ALLOWED_CATEGORIES:
            raise ValueError(f"category must be one of {ALLOWED_CATEGORIES}, got '{cat}'")
        diff = d.get("difficulty", "medium")
        if diff not in ALLOWED_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {ALLOWED_DIFFICULTIES}, got '{diff}'")
        return cls(
            id=d["id"],
            query=d["query"],
            paper_id=d.get("paper_id"),
            paper_name=d.get("paper_name"),
            gt_relevant_pages=d.get("gt_relevant_pages", []),
            gt_answer_facts=d.get("gt_answer_facts", []),
            gt_keywords=d.get("gt_keywords", []),
            category=cat,
            difficulty=diff,
        )


@dataclass
class EvalDataset:
    """评估数据集容器。"""

    name: str = "VRAG Eval Dataset"
    version: str = "1.0"
    cases: list[EvalCase] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path | str) -> "EvalDataset":
        """从 JSON 文件加载评估数据集。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = [EvalCase.from_dict(c) for c in data.get("cases", [])]
        return cls(
            name=data.get("name", "VRAG Eval Dataset"),
            version=data.get("version", "1.0"),
            cases=cases,
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def by_category(self, category: str) -> list[EvalCase]:
        return [c for c in self.cases if c.category == category]

    def by_difficulty(self, difficulty: str) -> list[EvalCase]:
        return [c for c in self.cases if c.difficulty == difficulty]
