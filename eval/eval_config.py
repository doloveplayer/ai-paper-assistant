"""评估系统全局配置 —— 阈值、路径、Judge 参数。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiagnosticThresholds:
    """闭环诊断阈值 —— 低于阈值的指标会触发诊断建议。"""
    recall_at_3: float = 0.5
    precision_at_3: float = 0.4
    mrr: float = 0.3
    faithfulness: float = 3.0
    answer_relevance: float = 3.5
    factual_correctness: float = 3.0
    token_efficiency_tps: float = 2.0


@dataclass
class EvalConfig:
    """评估系统全局配置。"""

    # ---- 路径 ----
    eval_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    dataset_path: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "eval_dataset.json")
    history_dir: Path = field(default_factory=lambda: Path(__file__).parent / "history")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).parent / "reports")

    # ---- LLM-as-a-Judge 配置 ----
    judge_api_base: str = "http://localhost:8000/v1"
    judge_model: str = "qwen2.5-7b-instruct"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 512
    judge_timeout: int = 60

    # ---- RAG 检索配置 ----
    top_k: int = 3
    pool_size_multiplier: int = 6

    # ---- 诊断阈值 ----
    thresholds: DiagnosticThresholds = field(default_factory=DiagnosticThresholds)

    # ---- 评估执行 ----
    thread_timeout_per_case: int = 120  # 单个用例最大等待秒数


# 全局单例
config = EvalConfig()
