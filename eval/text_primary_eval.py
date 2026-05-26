"""文本主路评估 — 对比新旧管线检索质量。

评估维度:
  Layer 1: 检索质量 (Recall@3, Precision@3, MRR, NDCG@3)
  Layer 2: 生成质量 (Faithfulness, Relevance, Correctness — LLM-as-a-Judge)

对比: 旧双路 (vrag_hybrid_collection) vs 新文本主路 (vrag_text_primary)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.eval_dataset import EvalDataset
from eval.eval_metrics import compute_retrieval_metrics
from eval.eval_judge import FaithfulnessJudge, AnswerRelevanceJudge, FactualCorrectnessJudge
from eval.eval_tracer import EvalContext
from config import Config

TOP_K = 3


def _build_judge_context(fused_hits: list[dict]) -> str:
    """从检索命中构建 Faithfulness judge 可用的文本上下文。

    包含：父块文本 + 子块文本 + 表格数据 + 视觉页面标记。
    Judge 虽然是纯文本模型，但表格 JSON 和图片描述可提供间接证据。
    """
    parts = []
    for h in fused_hits:
        p = h.get("payload", {})
        page = p.get("page_number", 0)
        is_visual = p.get("is_visual_page", False)

        header = f"[Page {page}]"
        if is_visual:
            header += " [含图表/表格]"

        lines = [header]

        parent_text = p.get("parent_text", "")
        if parent_text:
            lines.append(f"  父块文本: {parent_text[:1500]}")

        child_text = p.get("child_text", "")
        if child_text and child_text != parent_text:
            lines.append(f"  子块文本: {child_text[:800]}")

        tables = p.get("page_tables", {})
        if tables:
            lines.append(f"  表格数据: {json.dumps(tables, ensure_ascii=False)[:500]}")

        parts.append("\n".join(lines))

    return "\n\n---\n".join(parts)


def run_text_primary_eval():
    """对新文本主路集合跑全量 10 用例评估，对比旧双路基准数据。"""
    dataset_path = Path(__file__).parent / "data" / "eval_dataset.json"
    dataset = EvalDataset.from_json(str(dataset_path))

    faith_judge = FaithfulnessJudge()
    rel_judge = AnswerRelevanceJudge()
    cor_judge = FactualCorrectnessJudge()

    results = []
    print("=" * 70)
    print("文本主路评估 (text_primary — 父子分块 + 加权RRF)")
    print(f"集合: {Config.TEXT_PRIMARY_COLLECTION}")
    print("=" * 70)

    for case in dataset:
        case_id = case.id
        print(f"\n{'─' * 50}")
        print(f"📋 [{case_id}] {case.query[:80]}...")
        print(f"   paper_id: {case.paper_id}")

        t0 = time.time()

        # ---- 开启 Tracer 捕获实际检索命中 ----
        ctx = EvalContext.enable()

        try:
            from tools.text_primary_rag import search_text_primary_knowledge
            raw_result = search_text_primary_knowledge.invoke({
                "query": case.query,
                "paper_id": case.paper_id,
            })
            elapsed = time.time() - t0
            print(f"   🔍 检索完成 ({elapsed:.1f}s)")
            print(f"   结果前200字: {raw_result[:200]}...")

            # ---- 从 Tracer 提取实际检索命中 ----
            if ctx.has_retrieval_data:
                trace = ctx.last_retrieval
                vision_hits_raw = trace.vision_hits
                text_hits_raw = trace.text_hits
                fused_hits_raw = trace.fused_hits
                print(f"   📡 Tracer: vision={len(vision_hits_raw)} text={len(text_hits_raw)} fused={len(fused_hits_raw)}")
            else:
                print(f"   ⚠️ Tracer 未捕获到检索数据，回退到空列表")
                vision_hits_raw = []
                text_hits_raw = []
                fused_hits_raw = []
        except Exception as e:
            print(f"   ❌ 检索异常: {e}")
            results.append({
                "case_id": case_id, "query": case.query,
                "paper_id": case.paper_id, "error": str(e),
            })
            EvalContext.disable()
            continue
        finally:
            EvalContext.disable()

        # ---- 检索指标 (使用 tool 实际命中的数据) ----
        metrics = compute_retrieval_metrics(
            fused_hits=fused_hits_raw,
            gt_relevant_pages=case.gt_relevant_pages,
            vision_hits=vision_hits_raw,
            text_hits=text_hits_raw,
            top_k=TOP_K,
        )

        # ---- 生成质量 (Judge) ----
        body = raw_result.split("📚")[0] if "📚" in raw_result else raw_result[:1500]

        # 从 tracer 捕获的实际上下文构建 judge context
        if fused_hits_raw:
            context = _build_judge_context(fused_hits_raw)
        else:
            context = raw_result[:2000]

        try:
            faith = faith_judge.evaluate(case.query, body, context)
        except Exception:
            faith = type('obj', (object,), {'score': 0.0})()

        try:
            rel = rel_judge.evaluate(case.query, body)
        except Exception:
            rel = type('obj', (object,), {'score': 0.0})()

        try:
            cor = cor_judge.evaluate(
                body,
                gt_facts=case.gt_answer_facts,
                gt_facts_v2=case.gt_answer_facts_v2 if case.gt_answer_facts_v2 else None,
            )
        except Exception:
            cor = type('obj', (object,), {'score': 0.0, 'details': {}})()

        print(f"   📊 Recall@3={metrics.recall_at_k:.2f} "
              f"Prec@3={metrics.precision_at_k:.2f} "
              f"MRR={metrics.mrr:.2f} "
              f"Faith={faith.score:.0f} Rel={rel.score:.0f} Cor={cor.score:.0f}")

        results.append({
            "case_id": case_id,
            "query": case.query,
            "paper_id": case.paper_id,
            "category": case.category,
            "difficulty": case.difficulty,
            "retrieval": {
                "recall_at_3": metrics.recall_at_k,
                "precision_at_3": metrics.precision_at_k,
                "mrr": metrics.mrr,
                "ndcg_at_3": metrics.ndcg_at_k,
                "vision_recall": metrics.vision_recall,
                "text_recall": metrics.text_recall,
                "retrieved_pages": metrics.retrieved_pages,
                "relevant_pages": metrics.relevant_pages,
            },
            "generation": {
                "faithfulness": faith.score,
                "answer_relevance": rel.score,
                "factual_correctness": cor.score,
                "source_scores": getattr(cor, 'details', {}).get("source_scores", {}),
            },
        })

    # ---- 汇总 ----
    n = len([r for r in results if "error" not in r])
    if n == 0:
        print("\n❌ 无有效结果")
        return results

    avg_recall = sum(r["retrieval"]["recall_at_3"] for r in results if "error" not in r) / n
    avg_prec = sum(r["retrieval"]["precision_at_3"] for r in results if "error" not in r) / n
    avg_mrr = sum(r["retrieval"]["mrr"] for r in results if "error" not in r) / n
    avg_vision_recall = sum(
        (r["retrieval"].get("vision_recall") or 0) for r in results if "error" not in r
    ) / n
    avg_text_recall = sum(
        (r["retrieval"].get("text_recall") or 0) for r in results if "error" not in r
    ) / n
    avg_faith = sum(r["generation"]["faithfulness"] for r in results if "error" not in r) / n
    avg_rel = sum(r["generation"]["answer_relevance"] for r in results if "error" not in r) / n
    avg_cor = sum(r["generation"]["factual_correctness"] for r in results if "error" not in r) / n

    print("\n" + "=" * 70)
    print("文本主路评估汇总")
    print("=" * 70)
    print(f"完成用例: {n}/{len(dataset)}")
    print(f"Avg Recall@3:  {avg_recall:.2f}")
    print(f"Avg Precision@3: {avg_prec:.2f}")
    print(f"Avg MRR:       {avg_mrr:.2f}")
    print(f"Avg Vision Recall: {avg_vision_recall:.2f}")
    print(f"Avg Text Recall:   {avg_text_recall:.2f}")
    print(f"Avg Faithfulness:  {avg_faith:.1f}/5")
    print(f"Avg Relevance:     {avg_rel:.1f}/5")
    print(f"Avg Correctness:   {avg_cor:.1f}/5")

    # ---- 分来源 Correctness 统计 ----
    text_scores = []
    visual_scores = []
    for r in results:
        if "error" in r:
            continue
        ss = r["generation"].get("source_scores", {})
        if "text" in ss:
            text_scores.append(ss["text"]["match_rate"])
        # figure + table 合并为 visual
        fig = ss.get("figure", {}).get("match_rate", 0)
        tbl = ss.get("table", {}).get("match_rate", 0)
        fig_total = ss.get("figure", {}).get("total_facts", 0)
        tbl_total = ss.get("table", {}).get("total_facts", 0)
        if fig_total + tbl_total > 0:
            visual_scores.append((fig * fig_total + tbl * tbl_total) / (fig_total + tbl_total))

    if text_scores or visual_scores:
        print(f"\n{'─' * 40}")
        print("分来源 Correctness (match rate):")
        if text_scores:
            print(f"  文本类事实: {sum(text_scores)/len(text_scores):.2f}")
        if visual_scores:
            print(f"  图表类事实: {sum(visual_scores)/len(visual_scores):.2f}  ← text-only Judge 天然劣势")

    # ---- 与旧双路对比 ----
    print("\n" + "=" * 70)
    print("文本主路 vs 旧双路 对比")
    print("=" * 70)
    dual = {"recall": 0.57, "prec": 0.63, "mrr": 0.80, "faith": 3.5, "rel": 4.3, "cor": 3.1}
    print(f"{'指标':<20} {'文本主路':>8} {'旧双路':>8} {'变化':>8}")
    print("-" * 48)
    print(f"{'Recall@3':<20} {avg_recall:>8.2f} {dual['recall']:>8.2f} {avg_recall-dual['recall']:>+8.2f}")
    print(f"{'Precision@3':<20} {avg_prec:>8.2f} {dual['prec']:>8.2f} {avg_prec-dual['prec']:>+8.2f}")
    print(f"{'MRR':<20} {avg_mrr:>8.2f} {dual['mrr']:>8.2f} {avg_mrr-dual['mrr']:>+8.2f}")
    print(f"{'Faithfulness':<20} {avg_faith:>8.1f} {dual['faith']:>8.1f} {avg_faith-dual['faith']:>+8.1f}")
    print(f"{'Answer Relevance':<20} {avg_rel:>8.1f} {dual['rel']:>8.1f} {avg_rel-dual['rel']:>+8.1f}")
    print(f"{'Factual Correctness':<20} {avg_cor:>8.1f} {dual['cor']:>8.1f} {avg_cor-dual['cor']:>+8.1f}")

    # 保存结果
    out_path = Path(__file__).parent / "history" / "text_primary_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "text_primary",
        "collection": Config.TEXT_PRIMARY_COLLECTION,
        "avg_recall_at_3": avg_recall,
        "avg_precision_at_3": avg_prec,
        "avg_mrr": avg_mrr,
        "avg_vision_recall": avg_vision_recall,
        "avg_text_recall": avg_text_recall,
        "avg_faithfulness": avg_faith,
        "avg_answer_relevance": avg_rel,
        "avg_factual_correctness": avg_cor,
        "completed_cases": n,
        "total_cases": len(dataset),
        "dual_baseline_v4": dual,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")

    return results


if __name__ == "__main__":
    run_text_primary_eval()
