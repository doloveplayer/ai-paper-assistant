"""历史结果持久化与趋势对比。

存储结构:
    eval/history/
    ├── index.json              # 运行索引 [{run_id, timestamp, summary}]
    ├── 2026-04-27_143052.json  # 单次运行完整结果
    └── ...
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from eval.eval_config import config


class EvalStorage:
    """评估结果的持久化存储与历史查询。"""

    def __init__(self):
        self.history_dir = config.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.history_dir / "index.json"

    # -----------------------------------------------------------
    # 保存
    # -----------------------------------------------------------
    def save_run(self, results: list, dataset) -> Path:
        """保存单次运行的完整结果。返回结果文件路径。"""
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_id = ts
        file_path = self.history_dir / f"{run_id}.json"

        # 汇总
        valid_retrieval = [r for r in results if r.retrieval_metrics is not None]
        valid_faith = [r for r in results if r.faithfulness is not None]
        valid_rel = [r for r in results if r.answer_relevance is not None]

        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        payload = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset.name if dataset else "unknown",
            "num_cases": len(results),
            "summary": {
                "avg_recall_at_3": mean([r.retrieval_metrics.recall_at_k for r in valid_retrieval]),
                "avg_precision_at_3": mean([r.retrieval_metrics.precision_at_k for r in valid_retrieval]),
                "avg_mrr": mean([r.retrieval_metrics.mrr for r in valid_retrieval]),
                "avg_faithfulness": mean([r.faithfulness.score for r in valid_faith]),
                "avg_answer_relevance": mean([r.answer_relevance.score for r in valid_rel]),
                "errors": sum(1 for r in results if r.error is not None),
            },
            "results": [r.to_dict() for r in results],
        }

        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # 更新 index.json
        self._update_index(run_id, payload["summary"])

        print(f"Results saved to {file_path}")
        return file_path

    def _update_index(self, run_id: str, summary: dict):
        """在 index.json 中追加运行记录（保留最近 100 条）。"""
        if self.index_path.exists():
            index = json.loads(self.index_path.read_text())
        else:
            index = []

        index.append({"run_id": run_id, "timestamp": datetime.now().isoformat(), "summary": summary})

        # 仅保留最近 100 条
        if len(index) > 100:
            index = index[-100:]

        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # -----------------------------------------------------------
    # 查询
    # -----------------------------------------------------------
    def load_run(self, run_id: str) -> dict | None:
        """加载指定运行的全部结果。"""
        file_path = self.history_dir / f"{run_id}.json"
        if file_path.exists():
            return json.loads(file_path.read_text())
        return None

    def list_runs(self) -> list[dict]:
        """列出所有历史运行记录。"""
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return []

    def compare_runs(self, run_id1: str, run_id2: str) -> dict:
        """对比两次运行的指标变化。"""
        run1 = self.load_run(run_id1)
        run2 = self.load_run(run_id2)

        if not run1 or not run2:
            return {"error": "One or both runs not found"}

        s1 = run1["summary"]
        s2 = run2["summary"]

        comparison = {}
        for key in s1:
            if key in s2 and key != "errors":
                v1 = s1[key]
                v2 = s2[key]
                delta = v2 - v1
                comparison[key] = {
                    "before": v1,
                    "after": v2,
                    "delta": delta,
                    "trend": "up" if delta > 0 else "down" if delta < 0 else "flat",
                }

        return {
            "run1": run_id1,
            "run2": run_id2,
            "timestamp1": run1["timestamp"],
            "timestamp2": run2["timestamp"],
            "comparison": comparison,
        }

    def latest_trend(self, num_runs: int = 5) -> dict:
        """获取最近 N 次运行的指标趋势数据（供可视化）。"""
        runs = self.list_runs()[-num_runs:]
        if not runs:
            return {}

        trend = {"timestamps": [], "metrics": {}}
        for run in runs:
            trend["timestamps"].append(run["timestamp"][:16])
            for key, val in run["summary"].items():
                if key != "errors":
                    trend["metrics"].setdefault(key, []).append(val)

        return trend
