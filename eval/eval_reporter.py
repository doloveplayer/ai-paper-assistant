"""报告生成器 —— 终端彩色输出 + Markdown 报告文件 + 闭环诊断建议。"""

from datetime import datetime
from pathlib import Path

from eval.eval_config import config
from eval.eval_tracer import RetrievalTrace


class EvalReporter:
    """生成评估报告（终端 + Markdown 文件）。"""

    def __init__(self):
        self.reports_dir = config.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Markdown 报告
    # -----------------------------------------------------------
    def generate_markdown(self, results: list, dataset) -> str:
        """生成完整的 Markdown 评估报告。"""
        lines = []
        self._header(lines)
        self._aggregate_summary(lines, results)
        self._per_case_details(lines, results)
        self._diagnostics(lines, results)
        self._footer(lines)

        md = "\n".join(lines)

        # 保存到文件
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_path = self.reports_dir / f"eval_report_{ts}.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\nReport saved to {report_path}")

        return md

    def _header(self, lines: list[str]):
        lines.append("# VRAG RAG 闭环评估报告")
        lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**用例总数**: 见下方汇总\n")
        lines.append("---\n")

    # -----------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------
    def _aggregate_summary(self, lines: list[str], results: list):
        lines.append("## 汇总指标\n")

        n = len(results)
        valid_retrieval = [r for r in results if r.retrieval_metrics is not None]
        valid_faith = [r for r in results if r.faithfulness is not None]
        valid_rel = [r for r in results if r.answer_relevance is not None]
        valid_correct = [r for r in results if r.factual_correctness is not None]
        valid_latency = [r for r in results if r.error is None]

        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        lines.append("### 检索质量 (Retrieval Quality)\n")
        lines.append("| 指标 | 均值 | 说明 |")
        lines.append("|------|------|------|")
        if valid_retrieval:
            lines.append(f"| Recall@3 | {mean([r.retrieval_metrics.recall_at_k for r in valid_retrieval]):.2%} | 相关页面召回率 |")
            lines.append(f"| Precision@3 | {mean([r.retrieval_metrics.precision_at_k for r in valid_retrieval]):.2%} | 检索结果精确度 |")
            lines.append(f"| MRR | {mean([r.retrieval_metrics.mrr for r in valid_retrieval]):.3f} | 首个相关结果平均倒数排名 |")
            lines.append(f"| NDCG@3 | {mean([r.retrieval_metrics.ndcg_at_k for r in valid_retrieval]):.3f} | 排序质量 |")
            lines.append(f"| Hit Rate | {mean([r.retrieval_metrics.hit_rate for r in valid_retrieval]):.1%} | 至少命中 1 页的比例 |")
            v_recall = mean([r.retrieval_metrics.vision_recall for r in valid_retrieval if r.retrieval_metrics.vision_recall is not None])
            t_recall = mean([r.retrieval_metrics.text_recall for r in valid_retrieval if r.retrieval_metrics.text_recall is not None])
            if v_recall or t_recall:
                lines.append(f"| Vision Recall@3 | {v_recall:.2%} | ColPali 视觉路独立召回 |")
                lines.append(f"| Text Recall@3 | {t_recall:.2%} | BGE-M3 文本路独立召回 |")
        else:
            lines.append("| N/A | 无含 ground-truth 页面的用例 |")

        lines.append("\n### 生成质量 (Generation Quality)\n")
        lines.append("| 指标 | 均值 | 通过阈值 | 通过率 |")
        lines.append("|------|------|----------|--------|")
        lines.append(f"| Faithfulness | {mean([r.faithfulness.score for r in valid_faith]):.1f}/5 | >= {config.thresholds.faithfulness:.1f} | {sum(1 for r in valid_faith if r.faithfulness.passed)/max(len(valid_faith),1):.0%} |")
        lines.append(f"| Answer Relevance | {mean([r.answer_relevance.score for r in valid_rel]):.1f}/5 | >= {config.thresholds.answer_relevance:.1f} | {sum(1 for r in valid_rel if r.answer_relevance.passed)/max(len(valid_rel),1):.0%} |")
        lines.append(f"| Factual Correctness | {mean([r.factual_correctness.score for r in valid_correct]):.1f}/5 | >= {config.thresholds.factual_correctness:.1f} | {sum(1 for r in valid_correct if r.factual_correctness.passed)/max(len(valid_correct),1):.0%} |")

        lines.append("\n### 端到端效率 (End-to-End Efficiency)\n")
        lines.append("| 指标 | 均值 |")
        lines.append("|------|------|")
        lines.append(f"| Total Latency | {mean([r.total_latency for r in valid_latency]):.1f}s |")
        lines.append(f"| Input Tokens | {int(mean([r.total_input_tokens for r in valid_latency]))} |")
        lines.append(f"| Output Tokens | {int(mean([r.total_output_tokens for r in valid_latency]))} |")

    # -----------------------------------------------------------
    # 单用例详情
    # -----------------------------------------------------------
    def _per_case_details(self, lines: list[str], results: list):
        lines.append("\n---\n## 用例详情\n")

        for i, r in enumerate(results):
            lines.append(f"### [{r.case.id}] {r.case.query[:100]}")
            lines.append(f"- **类别**: {r.case.category} | **难度**: {r.case.difficulty}")
            if r.case.paper_id:
                lines.append(f"- **目标论文**: {r.case.paper_name or r.case.paper_id}")

            if r.error:
                lines.append(f"- **状态**:  FAIL ({r.error})\n")
                continue

            if r.retrieval_metrics:
                m = r.retrieval_metrics
                lines.append(f"- **检索**: Recall@3={m.recall_at_k:.2%}, Precision@3={m.precision_at_k:.2%}, MRR={m.mrr:.3f}, Hit={'YES' if m.hit_rate > 0 else 'NO'}")
                lines.append(f"  - Retrieved pages: {m.retrieved_pages} | GT pages: {m.relevant_pages} | Hits: {m.hit_pages}")
                if m.vision_recall is not None:
                    lines.append(f"  - Vision Recall: {m.vision_recall:.2%} | Text Recall: {m.text_recall:.2%}")

            judges = []
            if r.faithfulness:
                judges.append(f"Faithfulness={r.faithfulness.score:.0f}/5")
            if r.answer_relevance:
                judges.append(f"Relevance={r.answer_relevance.score:.0f}/5")
            if r.factual_correctness:
                judges.append(f"Correctness={r.factual_correctness.score:.0f}/5")
            if judges:
                lines.append(f"- **生成**: {' | '.join(judges)}")

            lines.append(f"- **效率**: {r.total_latency:.1f}s, in={r.total_input_tokens}, out={r.total_output_tokens}")

            # 显示 Judge 评语
            for j in [r.faithfulness, r.answer_relevance, r.factual_correctness]:
                if j and j.reasoning:
                    lines.append(f"  - *{j.metric_name}*: {j.reasoning[:200]}")

            lines.append("")

    # -----------------------------------------------------------
    # 闭环诊断
    # -----------------------------------------------------------
    def _diagnostics(self, lines: list[str], results: list):
        lines.append("---\n## 闭环诊断与改进建议\n")

        suggestions = self._analyze_results(results)
        if not suggestions:
            lines.append("所有指标均在阈值以上，系统运行良好。")
            return

        for severity, title, detail in suggestions:
            icon = {"critical": "red_circle", "warning": "yellow_circle", "info": "large_blue_circle"}.get(severity, "white_circle")
            lines.append(f"### {icon} {title}")
            lines.append(f"\n{detail}\n")

    def _analyze_results(self, results: list) -> list[tuple[str, str, str]]:
        """分析评估结果，生成诊断建议列表。返回 [(severity, title, detail)]。"""
        suggestions: list[tuple[str, str, str]] = []
        t = config.thresholds

        valid_retrieval = [r for r in results if r.retrieval_metrics is not None]
        valid_faith = [r for r in results if r.faithfulness is not None]
        valid_rel = [r for r in results if r.answer_relevance is not None]
        valid_correct = [r for r in results if r.factual_correctness is not None]
        valid_latency = [r for r in results if r.error is None]

        if not results:
            return [("info", "无数据", "没有可分析的评估结果。")]

        # 检索指标诊断
        if valid_retrieval:
            avg_recall = sum(r.retrieval_metrics.recall_at_k for r in valid_retrieval) / len(valid_retrieval)
            avg_precision = sum(r.retrieval_metrics.precision_at_k for r in valid_retrieval) / len(valid_retrieval)
            avg_mrr = sum(r.retrieval_metrics.mrr for r in valid_retrieval) / len(valid_retrieval)

            if avg_recall < t.recall_at_3:
                suggestions.append(("critical",
                    f"检索召回率过低 (Recall@3={avg_recall:.2%} < {t.recall_at_3:.0%})",
                    f"**问题**: 检索遗漏了 {len(valid_retrieval)} 个用例中的关键页面。\n\n"
                    f"**建议操作**:\n"
                    f"1. 检查 ColPali 是否完整索引了所有论文页面（`tools/mix_ingest.py` 断点续传）\n"
                    f"2. 尝试增大 `search_vision_knowledge` 中 `pool_size = top_k * multiplier` 的 multiplier（当前 {config.pool_size_multiplier} → 建议 10）\n"
                    f"3. 对英文术语较多的查询，尝试 query expansion（同义词扩展）\n"
                    f"4. 检查 BGE-M3 文本路是否因中英文混合查询导致召回偏低"))

            if avg_precision < t.precision_at_3:
                suggestions.append(("warning",
                    f"检索精确度偏低 (Precision@3={avg_precision:.2%} < {t.precision_at_3:.0%})",
                    f"**问题**: 检索返回了过多不相关页面，噪声影响 VLM 分析质量。\n\n"
                    f"**建议操作**:\n"
                    f"1. 调整 RRF 融合参数 k 值（当前 60），减小 k 让排名更敏感 → 建议 40\n"
                    f"2. 在 `search_vision_knowledge` 中加强 paper_id 过滤\n"
                    f"3. 考虑给文本路和视觉路不同的 RRF 权重（当前等权）"))

            if avg_mrr < t.mrr:
                suggestions.append(("warning",
                    f"首个相关结果排名靠后 (MRR={avg_mrr:.3f} < {t.mrr})",
                    f"**建议操作**:\n"
                    f"1. 增大 ColPali 的 query token 数量以提升视觉路精度\n"
                    f"2. 考虑在 RRF 融合前对两路结果做 score normalization"))

        # 生成质量诊断
        if valid_faith:
            avg_faith = sum(r.faithfulness.score for r in valid_faith) / len(valid_faith)
            if avg_faith < t.faithfulness:
                suggestions.append(("critical",
                    f"忠实度过低 —— 存在幻觉风险 (Faithfulness={avg_faith:.1f}/5 < {t.faithfulness:.0f})",
                    f"**问题**: 生成的回答中包含无法在检索上下文中验证的事实。\n\n"
                    f"**建议操作**:\n"
                    f"1. 降低 VLM temperature（当前 0.3 → 建议 0.1），减少随机生成\n"
                    f"2. 在 `_summarize_payloads` 中保留更多原文细节（当前 truncate 3000 字/页）\n"
                    f"3. 强化 system prompt 中的护栏规则（`agent_core.py` 第 131 行）\n"
                    f"4. 检查 `_summarize_payloads` 压缩是否丢失了关键信息"))

        if valid_rel:
            avg_rel = sum(r.answer_relevance.score for r in valid_rel) / len(valid_rel)
            if avg_rel < t.answer_relevance:
                suggestions.append(("warning",
                    f"回答相关性偏低 (Relevance={avg_rel:.1f}/5 < {t.answer_relevance:.0f})",
                    f"**建议操作**:\n"
                    f"1. 在 system prompt 中增加聚焦指令，要求 VLM 严格围绕 query 回答\n"
                    f"2. 检查 VLM prompt 中的 `【护栏规则】` 是否被遵守"))

        if valid_correct:
            avg_correct = sum(r.factual_correctness.score for r in valid_correct) / len(valid_correct)
            if avg_correct < t.factual_correctness:
                suggestions.append(("critical",
                    f"事实正确性不足 (Correctness={avg_correct:.1f}/5 < {t.factual_correctness:.0f})",
                    f"**问题**: 回答中的关键事实与 ground truth 不一致。\n\n"
                    f"**建议操作**:\n"
                    f"1. 检查 `_summarize_payloads` 的 prompt 是否引入了偏差\n"
                    f"2. 考虑对 VLM 的输出做 post-check（如要求引用具体页码）\n"
                    f"3. 检查 ColPali 检索到的页面是否真正包含目标信息"))

        # 效率诊断
        if valid_latency:
            avg_lat = sum(r.total_latency for r in valid_latency) / len(valid_latency)
            avg_tps = sum(r.total_output_tokens for r in valid_latency) / max(sum(r.total_latency for r in valid_latency), 0.001)
            if avg_tps < t.token_efficiency_tps:
                suggestions.append(("info",
                    f"推理效率偏低 (Throughput={avg_tps:.1f} t/s < {t.token_efficiency_tps})",
                    f"**建议操作**:\n"
                    f"1. 检查 vLLM 的 continuous batching 是否正常工作\n"
                    f"2. 增大 `--gpu-memory-utilization` 以支持更大的 KV cache\n"
                    f"3. 检查是否有其他进程抢占 GPU"))

        return suggestions

    def _footer(self, lines: list[str]):
        lines.append("\n---\n")
        lines.append("*报告由 VRAG 闭环评估系统自动生成。阈值配置见 `eval/eval_config.py`。*")
