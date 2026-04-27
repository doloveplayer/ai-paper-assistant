"""检索指标计算 —— Recall@k, Precision@k, MRR, NDCG@k, Hit Rate。

支持分别计算视觉路、文本路、RRF 融合后的指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class RetrievalMetrics:
    """单次查询的检索质量指标。"""

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0

    # 分路指标
    vision_recall: float | None = None
    text_recall: float | None = None
    fused_recall: float | None = None

    # 诊断信息
    retrieved_pages: list[int] = field(default_factory=list)
    relevant_pages: list[int] = field(default_factory=list)
    hit_pages: list[int] = field(default_factory=list)


def _extract_page_numbers(hits: list[dict]) -> list[int]:
    """从 hit 列表中提取页码（去重、排序）。"""
    pages = []
    for h in hits:
        p = h.get("payload", {}).get("page_number")
        if p is not None:
            pages.append(int(p))
    return sorted(set(pages))


def _dcg(scores: list[float]) -> float:
    """Discounted Cumulative Gain。"""
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def compute_retrieval_metrics(
    fused_hits: list[dict],
    gt_relevant_pages: list[int],
    vision_hits: list[dict] | None = None,
    text_hits: list[dict] | None = None,
    top_k: int = 3,
) -> RetrievalMetrics:
    """计算完整的检索质量指标。

    Args:
        fused_hits: RRF 融合后的检索结果（ScoredPoint dict 列表）
        gt_relevant_pages: Ground-truth 相关页码
        vision_hits: 视觉路独立检索结果（可选，用于分路对比）
        text_hits: 文本路独立检索结果（可选，用于分路对比）
        top_k: 截断值

    Returns:
        RetrievalMetrics 包含所有指标和诊断信息
    """
    gt_set = set(gt_relevant_pages)

    # ---- RRF 融合路指标 ----
    retrieved_pages = _extract_page_numbers(fused_hits[:top_k])
    retrieved_set = set(retrieved_pages)
    hit_set = retrieved_set & gt_set

    # Recall@k
    recall = len(hit_set) / len(gt_set) if gt_set else 0.0

    # Precision@k
    precision = len(hit_set) / len(retrieved_set) if retrieved_set else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, h in enumerate(fused_hits[:top_k]):
        page = int(h.get("payload", {}).get("page_number", -1))
        if page in gt_set:
            mrr = 1.0 / (rank + 1)
            break

    # NDCG@k: 相关性二元 (1 if relevant, 0 otherwise)
    ideal_scores = sorted([1.0] * min(len(gt_set), top_k) + [0.0] * max(0, top_k - len(gt_set)), reverse=True)
    actual_scores = [1.0 if p in gt_set else 0.0 for p in retrieved_pages]
    ndcg = _dcg(actual_scores) / _dcg(ideal_scores) if _dcg(ideal_scores) > 0 else 0.0

    # Hit Rate
    hit_rate = 1.0 if hit_set else 0.0

    metrics = RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        ndcg_at_k=ndcg,
        hit_rate=hit_rate,
        retrieved_pages=retrieved_pages,
        relevant_pages=sorted(gt_set),
        hit_pages=sorted(hit_set),
    )

    # ---- 分路指标（仅 Recall）----
    if vision_hits is not None:
        v_pages = _extract_page_numbers(vision_hits[:top_k])
        metrics.vision_recall = len(set(v_pages) & gt_set) / len(gt_set) if gt_set else 0.0

    if text_hits is not None:
        t_pages = _extract_page_numbers(text_hits[:top_k])
        metrics.text_recall = len(set(t_pages) & gt_set) / len(gt_set) if gt_set else 0.0

    return metrics


def compute_aggregated_metrics(all_metrics: list[RetrievalMetrics]) -> dict:
    """汇总多次查询的指标（均值）。"""
    n = len(all_metrics)
    if n == 0:
        return {}

    def mean(vals):
        return sum(vals) / len(vals)

    return {
        "avg_recall_at_3": mean([m.recall_at_k for m in all_metrics]),
        "avg_precision_at_3": mean([m.precision_at_k for m in all_metrics]),
        "avg_mrr": mean([m.mrr for m in all_metrics]),
        "avg_ndcg_at_3": mean([m.ndcg_at_k for m in all_metrics]),
        "hit_rate": mean([m.hit_rate for m in all_metrics]),
        "avg_vision_recall": mean([m.vision_recall for m in all_metrics if m.vision_recall is not None]) if any(m.vision_recall is not None for m in all_metrics) else None,
        "avg_text_recall": mean([m.text_recall for m in all_metrics if m.text_recall is not None]) if any(m.text_recall is not None for m in all_metrics) else None,
        "num_queries": n,
    }
