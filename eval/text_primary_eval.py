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
from config import Config

TOP_K = 3


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

        # ---- 调用文本主路检索工具 ----
        try:
            from tools.text_primary_rag import search_text_primary_knowledge
            raw_result = search_text_primary_knowledge.invoke({
                "query": case.query,
                "paper_id": case.paper_id,
            })
            elapsed = time.time() - t0
            print(f"   🔍 检索完成 ({elapsed:.1f}s)")
            print(f"   结果前200字: {raw_result[:200]}...")
        except Exception as e:
            print(f"   ❌ 检索异常: {e}")
            # Try to get last retrieval hits
            raw_result = f"[ERROR: {e}]"
            results.append({
                "case_id": case_id, "query": case.query,
                "paper_id": case.paper_id, "error": str(e),
            })
            continue

        # ---- 提取检索命中 ----
        try:
            from tools.text_primary_rag import _client, COLLECTION_NAME
            if _client is None:
                raise RuntimeError("检索未初始化 Qdrant client")

            # 重建检索命中的数据
            import qdrant_client as qc
            from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchText
            from sentence_transformers import SentenceTransformer

            text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")
            text_query = text_model.encode(case.query, normalize_embeddings=True).tolist()

            pid = case.paper_id
            pid_variants = [pid]
            if pid.endswith(".pdf"):
                pid_variants.append(pid[:-4])
            else:
                pid_variants.append(pid + ".pdf")
            clean = pid.strip().removesuffix(".pdf").replace("\n", " ")
            prefix = clean[:40] if len(clean) > 40 else clean
            qf = Filter(should=[
                FieldCondition(key="paper_id", match=MatchAny(any=pid_variants)),
                FieldCondition(key="file_name", match=MatchAny(any=pid_variants)),
                FieldCondition(key="file_name", match=MatchText(text=prefix)),
            ])

            # 文本路
            text_res = _client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_query,
                using="text_dense",
                query_filter=qf,
                limit=Config.TEXT_SEARCH_POOL_SIZE,
            ).points

            # 去重
            parent_best = {}
            for h in text_res:
                pid_val = h.payload.get("parent_id", h.id)
                if pid_val not in parent_best or h.score > parent_best[pid_val].score:
                    parent_best[pid_val] = h
            deduped = sorted(parent_best.values(), key=lambda h: h.score, reverse=True)[:TOP_K]

            text_hits_raw = [{"id": h.id, "score": h.score, "payload": dict(h.payload)} for h in list(parent_best.values())]
            fused_hits_raw = [{"id": h.id, "score": h.score, "payload": dict(h.payload)} for h in deduped]

        except Exception as e:
            print(f"   ⚠️ 检索指标提取失败: {e}")
            text_hits_raw = []
            fused_hits_raw = []

        # ---- 检索指标 ----
        metrics = compute_retrieval_metrics(
            fused_hits=fused_hits_raw,
            gt_relevant_pages=case.gt_relevant_pages,
            vision_hits=None,
            text_hits=text_hits_raw,
            top_k=TOP_K,
        )

        # ---- 生成质量 (Judge) ----
        # 从 raw_result 提取 body 和 context
        body = raw_result.split("📚")[0] if "📚" in raw_result else raw_result[:1500]
        context = "\n\n".join(
            f"[Page {h['payload'].get('page_number',0)}] {h['payload'].get('parent_text','')[:1000]}"
            for h in fused_hits_raw
        ) if fused_hits_raw else raw_result[:2000]

        try:
            faith = faith_judge.evaluate(case.query, body, context)
        except Exception:
            faith = type('obj', (object,), {'score': 0.0})()

        try:
            rel = rel_judge.evaluate(case.query, body)
        except Exception:
            rel = type('obj', (object,), {'score': 0.0})()

        try:
            cor = cor_judge.evaluate(body, case.gt_answer_facts)
        except Exception:
            cor = type('obj', (object,), {'score': 0.0})()

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
                "text_recall": metrics.text_recall,
                "retrieved_pages": metrics.retrieved_pages,
                "relevant_pages": metrics.relevant_pages,
            },
            "generation": {
                "faithfulness": faith.score,
                "answer_relevance": rel.score,
                "factual_correctness": cor.score,
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
    avg_text_recall = sum(r["retrieval"].get("text_recall", 0) or 0 for r in results if "error" not in r) / n
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
    print(f"Avg Text Recall: {avg_text_recall:.2f}")
    print(f"Avg Faithfulness:  {avg_faith:.1f}/5")
    print(f"Avg Relevance:     {avg_rel:.1f}/5")
    print(f"Avg Correctness:   {avg_cor:.1f}/5")

    # ---- 与旧双路对比 ----
    print("\n" + "=" * 70)
    print("文本主路 vs 旧双路 对比")
    print("=" * 70)
    # v4 最终双路结果 (硬编码基准)
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
