"""VRAG RAG 闭环评估系统。

提供完整的数据采集 → 指标计算 → LLM-as-a-Judge → 诊断报告 → 历史追踪管线。

主要接口:
    run_evaluation()          — 一键运行完整评估
    EvalRunner                — 评估编排器（可编程控制）
    RetrievalMetrics          — 检索指标计算
    FaithfulnessJudge et al.  — LLM-as-a-Judge 生成质量评估
    EvalStorage               — 历史结果持久化与趋势对比
"""

from eval.eval_config import EvalConfig
from eval.eval_tracer import EvalContext, RetrievalTrace, TelemetryTrace
from eval.eval_dataset import EvalDataset, EvalCase
from eval.eval_metrics import RetrievalMetrics, compute_retrieval_metrics
from eval.eval_judge import (
    FaithfulnessJudge,
    AnswerRelevanceJudge,
    FactualCorrectnessJudge,
    JudgeResult,
)
from eval.eval_runner import EvalRunner, run_evaluation
from eval.eval_reporter import EvalReporter
from eval.eval_storage import EvalStorage

__all__ = [
    "EvalConfig",
    "EvalContext",
    "RetrievalTrace",
    "TelemetryTrace",
    "EvalDataset",
    "EvalCase",
    "RetrievalMetrics",
    "compute_retrieval_metrics",
    "FaithfulnessJudge",
    "AnswerRelevanceJudge",
    "FactualCorrectnessJudge",
    "JudgeResult",
    "EvalRunner",
    "run_evaluation",
    "EvalReporter",
    "EvalStorage",
]
