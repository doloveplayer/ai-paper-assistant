"""评估编排器 —— 驱动真实 Agent → 收集数据 → 计算指标 → 运行 Judge → 诊断。

核心流程:
    for each eval case:
        1. 启用 EvalContext 线程局部追踪
        2. 通过 agent_core.app.stream() 驱动真实 Agent
        3. 收集 retrieval traces + telemetry traces + final answer
        4. 计算检索指标 (Recall/Precision/MRR/NDCG/Hit Rate)
        5. 运行 LLM-as-a-Judge (Faithfulness/Relevance/Correctness)
        6. 汇总单用例结果
    end for
    汇总全局指标 → 生成诊断报告 → 持久化历史
"""

from __future__ import annotations

import gc
import sys
import time
import uuid
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from eval.eval_config import config
from eval.eval_dataset import EvalDataset, EvalCase
from eval.eval_metrics import compute_retrieval_metrics, RetrievalMetrics
from eval.eval_judge import (
    FaithfulnessJudge,
    AnswerRelevanceJudge,
    FactualCorrectnessJudge,
    JudgeResult,
)
from eval.eval_reporter import EvalReporter
from eval.eval_storage import EvalStorage


class CaseResult:
    """单个评估用例的完整结果。"""

    def __init__(self, case: EvalCase):
        self.case = case
        self.retrieval_metrics: RetrievalMetrics | None = None
        self.faithfulness: JudgeResult | None = None
        self.answer_relevance: JudgeResult | None = None
        self.factual_correctness: JudgeResult | None = None
        self.final_answer: str = ""
        self.total_latency: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.error: str | None = None
        self.pages_context: str = ""  # 检索上下文的文本拼接

    @property
    def all_judges_passed(self) -> bool:
        judges = [self.faithfulness, self.answer_relevance, self.factual_correctness]
        return all(j.passed for j in judges if j is not None)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case.id,
            "query": self.case.query,
            "paper_id": self.case.paper_id,
            "category": self.case.category,
            "difficulty": self.case.difficulty,
            "retrieval": {
                "recall_at_3": self.retrieval_metrics.recall_at_k if self.retrieval_metrics else None,
                "precision_at_3": self.retrieval_metrics.precision_at_k if self.retrieval_metrics else None,
                "mrr": self.retrieval_metrics.mrr if self.retrieval_metrics else None,
                "ndcg_at_3": self.retrieval_metrics.ndcg_at_k if self.retrieval_metrics else None,
                "hit_rate": self.retrieval_metrics.hit_rate if self.retrieval_metrics else None,
                "vision_recall": self.retrieval_metrics.vision_recall if self.retrieval_metrics else None,
                "text_recall": self.retrieval_metrics.text_recall if self.retrieval_metrics else None,
                "retrieved_pages": self.retrieval_metrics.retrieved_pages if self.retrieval_metrics else [],
                "relevant_pages": self.retrieval_metrics.relevant_pages if self.retrieval_metrics else [],
            },
            "generation": {
                "faithfulness": self.faithfulness.score if self.faithfulness else None,
                "faithfulness_reasoning": self.faithfulness.reasoning if self.faithfulness else "",
                "answer_relevance": self.answer_relevance.score if self.answer_relevance else None,
                "answer_relevance_reasoning": self.answer_relevance.reasoning if self.answer_relevance else "",
                "factual_correctness": self.factual_correctness.score if self.factual_correctness else None,
                "factual_correctness_reasoning": self.factual_correctness.reasoning if self.factual_correctness else "",
            },
            "efficiency": {
                "total_latency_s": self.total_latency,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            },
            "final_answer": self.final_answer[:2000],
            "error": self.error,
        }


class EvalRunner:
    """评估编排器。

    用法:
        runner = EvalRunner()
        results = runner.run_all()          # 运行全部用例
        results = runner.run_filtered(      # 按条件筛选运行
            categories=["methodology"],
            difficulty="hard",
        )
    """

    def __init__(self):
        self.dataset: EvalDataset | None = None
        self.results: list[CaseResult] = []
        self.faithfulness_judge = FaithfulnessJudge()
        self.relevance_judge = AnswerRelevanceJudge()
        self.correctness_judge = FactualCorrectnessJudge()
        self.reporter = EvalReporter()
        self.storage = EvalStorage()

    # -----------------------------------------------------------
    # 数据加载
    # -----------------------------------------------------------
    def load_dataset(self, path: str | None = None):
        path = path or str(config.dataset_path)
        self.dataset = EvalDataset.from_json(path)
        print(f"Loaded {len(self.dataset)} eval cases from {path}")

    # -----------------------------------------------------------
    # 受限 Agent（仅检索，禁止下载/网页搜索以防 OOM）
    # -----------------------------------------------------------
    def _get_eval_app(self):
        """创建仅含 search_vision_knowledge 的受限 Agent。

        与正式 Agent (agent_core.app) 的区别：
        - 工具：仅 search_vision_knowledge（无 search_academic_papers / download_and_ingest_vision_paper）
        - 系统提示词：仅从已有知识库检索，不触发下载流程
        - 无记忆压缩节点（每个评估用例用独立 thread_id，上下文极短）

        GPU 内存管理：复用 agent_core 已加载的 llm（ChatOpenAI HTTP 客户端本身不占 GPU），
        避免重复初始化模型。仅 search_vision_knowledge 工具内部会按需懒加载 ColPali/BGE-M3。
        """
        if hasattr(self, '_eval_app'):
            return self._eval_app

        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage
        from tools.vision_rag import search_vision_knowledge
        from agent_core import AgentState
        from config import Config
        import sqlite3
        import os

        eval_tools = [search_vision_knowledge]
        eval_llm = ChatOpenAI(
            base_url=Config.LLM_API_BASE,
            api_key="EMPTY",
            model=Config.LLM_MODEL_NAME,
        )
        eval_llm_with_tools = eval_llm.bind_tools(eval_tools)

        EVAL_SYSTEM_PROMPT = (
            "You are an academic research assistant being evaluated on RAG quality.\n\n"
            "【CRITICAL - EVALUATION MODE RULES】:\n"
            "1. You have ONE tool: `search_vision_knowledge`. You MUST call it FIRST before giving any answer.\n"
            "2. NEVER skip tool calling — even if the query seems generic. Call the tool first, then answer.\n"
            "3. When calling search_vision_knowledge, ALWAYS pass the paper_id if the user message specifies one.\n"
            "4. Answer questions based SOLELY on what search_vision_knowledge returns.\n"
            "5. If the database truly lacks relevant information, state so — but ONLY after calling the tool.\n"
            "6. Use elegant Markdown formatting. Cite your sources at the end.\n"
            "7. DO NOT say '好的', '我将为您', '请稍等'. Answer directly.\n"
        )

        def call_eval_model(state):
            messages = state["messages"]
            sys_msg = SystemMessage(content=EVAL_SYSTEM_PROMPT)
            invoke_messages = [sys_msg] + messages

            t_start = time.time()
            response = eval_llm_with_tools.invoke(invoke_messages)
            e2e_latency = time.time() - t_start

            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            tpot = (e2e_latency / output_tokens) if output_tokens > 0 else 0
            throughput = (output_tokens / e2e_latency) if e2e_latency > 0 else 0

            print(f"📊 [Eval] E2E: {e2e_latency:.2f}s | in: {input_tokens} | out: {output_tokens}")

            # 评估遥测捕获
            try:
                from eval.eval_tracer import EvalContext
                ctx = EvalContext.get()
                if ctx and ctx.enabled:
                    ctx.capture_telemetry(
                        e2e_latency=e2e_latency,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        tpot=tpot,
                        throughput=throughput,
                    )
            except Exception:
                pass

            return {"messages": [response]}

        def should_continue(state) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "continue"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_eval_model)
        workflow.add_node("tools", ToolNode(eval_tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
        workflow.add_edge("tools", "agent")

        os.makedirs("chat_history", exist_ok=True)
        db_path = os.path.join("chat_history", "eval_agent_memory.sqlite")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)

        self._eval_app = workflow.compile(checkpointer=memory)
        return self._eval_app

    # -----------------------------------------------------------
    # 运行单个用例（后台线程 + 队列）
    # -----------------------------------------------------------
    def _run_single_case(self, case: EvalCase) -> CaseResult:
        """在独立线程中使用 EvalContext 运行 Agent。

        关键：EvalContext 是 thread-local 的，必须在 agent 所在的线程内启用。
        通过共享字典 shared_state 将数据传回主线程。
        """
        result = CaseResult(case)
        event_queue: queue.Queue = queue.Queue()
        shared_state: dict = {}  # 跨线程共享容器

        def _run_agent():
            from eval.eval_tracer import EvalContext
            # 在 agent 线程内启用追踪
            ctx = EvalContext.enable()
            shared_state["ctx_ready"] = True

            try:
                app = self._get_eval_app()
                # Pass paper_id in the query so the agent can filter search_vision_knowledge
                query_with_context = case.query
                if case.paper_id:
                    query_with_context = (
                        f"【CRITICAL: When calling search_vision_knowledge, you MUST pass "
                        f"paper_id EXACTLY as: '{case.paper_id}'. "
                        f"Do NOT modify, add .pdf, or change this string in any way.】\n\n"
                        f"{case.query}"
                    )
                inputs = {"messages": [HumanMessage(content=query_with_context)]}
                thread_cfg = {"configurable": {"thread_id": f"eval_{case.id}_{uuid.uuid4().hex[:8]}"}}

                t_start = time.time()
                for event in app.stream(inputs, config=thread_cfg, stream_mode="updates"):
                    # Also capture raw tool outputs for retrieval metrics
                    for node_name, node_state in event.items():
                        if node_name == "tools":
                            msgs = node_state.get("messages", [])
                            for m in msgs:
                                if hasattr(m, "content") and hasattr(m, "name"):
                                    event_queue.put(("tool_output", {"name": m.name, "content": str(m.content)[:5000]}))
                    event_queue.put(("event", event))
                event_queue.put(("done", t_start))
            except Exception as e:
                event_queue.put(("error", str(e)))
            finally:
                # 将 context 数据传回共享容器
                shared_state["retrieval_traces"] = ctx.retrieval_traces
                shared_state["telemetry_traces"] = ctx.telemetry_traces
                shared_state["agent_steps"] = ctx.agent_steps
                EvalContext.disable()

        agent_thread = threading.Thread(target=_run_agent, daemon=True)
        agent_thread.start()
        agent_thread.join(timeout=config.thread_timeout_per_case)

        # 从共享容器中取出 EvalContext 数据
        retrieval_traces: list = shared_state.get("retrieval_traces", [])
        telemetry_traces: list = shared_state.get("telemetry_traces", [])

        # 收集 stream 事件中的最终回答
        final_answer = ""
        tool_outputs: list[dict] = []  # Fallback: extract retrieval data from tool outputs
        while not event_queue.empty():
            try:
                msg_type, data = event_queue.get_nowait()
            except queue.Empty:
                break

            if msg_type == "done":
                t_start = data
                result.total_latency = time.time() - t_start
                break
            elif msg_type == "error":
                result.error = data
                break
            elif msg_type == "tool_output":
                tool_outputs.append(data)
            elif msg_type == "event":
                for node_name, node_state in data.items():
                    if node_name == "agent":
                        last_msg = node_state["messages"][-1]
                        if hasattr(last_msg, "content") and last_msg.content and not getattr(last_msg, "tool_calls", None):
                            final_answer = last_msg.content

        if result.error and not final_answer:
            result.final_answer = final_answer

        result.final_answer = final_answer

        # 从事件中收集剩余数据（如果 agent 还未完成）
        if not result.error and not result.total_latency:
            while True:
                try:
                    msg_type, data = event_queue.get(timeout=5)
                except queue.Empty:
                    result.error = "Evaluation timed out waiting for agent completion"
                    break

                if msg_type == "done":
                    t_start = data
                    result.total_latency = time.time() - t_start
                    break
                elif msg_type == "error":
                    result.error = data
                    break
                elif msg_type == "event":
                    for node_name, node_state in data.items():
                        if node_name == "agent":
                            last_msg = node_state["messages"][-1]
                            if hasattr(last_msg, "content") and last_msg.content and not getattr(last_msg, "tool_calls", None):
                                final_answer = last_msg.content

        result.final_answer = result.final_answer or final_answer

        # 收集遥测
        result.total_input_tokens = sum(t.input_tokens for t in telemetry_traces)
        result.total_output_tokens = sum(t.output_tokens for t in telemetry_traces)

        # ---- 计算检索指标 ----
        # 方案 A: EvalContext capture（正常路径）
        # 方案 B: 从 vision_rag 全局缓存读取（绕过 threading.local 限制）
        if not retrieval_traces and case.gt_relevant_pages:
            try:
                from tools.vision_rag import _last_retrieval_hits
                hits = _last_retrieval_hits
                if hits and hits.get("fused_hits"):
                    from eval.eval_tracer import RetrievalTrace
                    retrieval_traces.append(RetrievalTrace(
                        query=case.query,
                        paper_id=case.paper_id,
                        vision_hits=hits.get("vision_hits", []),
                        text_hits=hits.get("text_hits", []),
                        fused_hits=hits.get("fused_hits", []),
                        final_pages_info=[],
                        final_answer="",
                    ))
            except Exception:
                pass

        if retrieval_traces and case.gt_relevant_pages:
            t = retrieval_traces[-1]
            result.retrieval_metrics = compute_retrieval_metrics(
                fused_hits=t.fused_hits,
                gt_relevant_pages=case.gt_relevant_pages,
                vision_hits=t.vision_hits,
                text_hits=t.text_hits,
                top_k=config.top_k,
            )

        # ---- 运行 LLM-as-a-Judge ----
        if result.final_answer:
            # Faithfulness: 需要检索上下文
            context_text = self._build_context_from_traces(retrieval_traces)
            if context_text:
                result.faithfulness = self.faithfulness_judge.evaluate(
                    query=case.query,
                    answer=result.final_answer,
                    context=context_text,
                )

            # Answer Relevance
            result.answer_relevance = self.relevance_judge.evaluate(
                query=case.query,
                answer=result.final_answer,
            )

            # Factual Correctness: 需要 ground truth facts
            if case.gt_answer_facts:
                result.factual_correctness = self.correctness_judge.evaluate(
                    answer=result.final_answer,
                    gt_facts=case.gt_answer_facts,
                )

        # ---- 用例完成后的内存清理（防止 GPU 碎片化累积导致 OOM） ----
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return result

    def _build_context_from_traces(self, retrieval_traces: list) -> str:
        """从检索轨迹重建上下文文本（供 Judge 使用）。"""
        parts = []
        for trace in retrieval_traces:
            for h in trace.fused_hits:
                payload = h.get("payload", {})
                doc_id = payload.get("paper_id") or payload.get("file_name") or "unknown"
                page_num = payload.get("page_number", "?")
                page_text = payload.get("page_text", "")
                if page_text:
                    parts.append(f"[{doc_id} | page {page_num}] {page_text[:1000]}")
        return "\n\n".join(parts[:5])




    # -----------------------------------------------------------
    # 批量运行
    # -----------------------------------------------------------
    def _prewarm_models(self):
        """预加载 ColPali + BGE-M3，避免第一个用例因懒加载和 GPU 碎片化导致 OOM/超时。"""
        import torch, gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        from tools.vision_rag import _get_vision_model, _get_text_model
        print("🔥 [预加载] 正在预热 ColPali + BGE-M3 ...")
        try:
            _get_vision_model()   # ColPali INT8 → GPU (~4GB, 一次性)
            print("   ColPali ✓")
        except Exception as e:
            print(f"   ColPali ✗ ({e}) — 将继续,但首用例可能失败")
        try:
            _get_text_model()     # BGE-M3 → CPU
            print("   BGE-M3 ✓")
        except Exception as e:
            print(f"   BGE-M3 ✗ ({e})")

    def run_all(self) -> list[CaseResult]:
        if self.dataset is None:
            self.load_dataset()

        # 预加载 ColPali + BGE-M3，避免第一个用例因懒加载超时/OOM
        self._prewarm_models()

        self.results = []
        for i, case in enumerate(self.dataset.cases):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(self.dataset)}] {case.id}: {case.query[:80]}...")
            print(f"     Category: {case.category} | Difficulty: {case.difficulty}")
            result = self._run_single_case(case)
            self.results.append(result)
            self._print_case_summary(result)

        return self.results

    def run_filtered(
        self,
        categories: list[str] | None = None,
        difficulty: str | None = None,
    ) -> list[CaseResult]:
        if self.dataset is None:
            self.load_dataset()

        cases = self.dataset.cases
        if categories:
            cases = [c for c in cases if c.category in categories]
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]

        self.results = []
        for i, case in enumerate(cases):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(cases)}] {case.id}: {case.query[:80]}...")
            result = self._run_single_case(case)
            self.results.append(result)
            self._print_case_summary(result)

        return self.results

    def _print_case_summary(self, result: CaseResult):
        """打印单个用例的简要结果。"""
        if result.error:
            print(f"  FAIL  | Error: {result.error}")
            return

        parts = []
        if result.retrieval_metrics:
            m = result.retrieval_metrics
            parts.append(f"Recall@3={m.recall_at_k:.2f}")
            parts.append(f"Precision@3={m.precision_at_k:.2f}")
            parts.append(f"MRR={m.mrr:.2f}")
        else:
            parts.append("Retrieval: N/A (no gt_pages)")

        if result.faithfulness:
            parts.append(f"Faith={result.faithfulness.score:.0f}/5")
        if result.answer_relevance:
            parts.append(f"Relevance={result.answer_relevance.score:.0f}/5")
        if result.factual_correctness:
            parts.append(f"Correct={result.factual_correctness.score:.0f}/5")

        parts.append(f"Latency={result.total_latency:.1f}s")
        print(f"  OK    | {' | '.join(parts)}")

    # -----------------------------------------------------------
    # 报告与存储
    # -----------------------------------------------------------
    def generate_report(self) -> str:
        return self.reporter.generate_markdown(self.results, self.dataset)

    def save_results(self) -> Path:
        return self.storage.save_run(self.results, self.dataset)


def run_evaluation(
    dataset_path: str | None = None,
    categories: list[str] | None = None,
    difficulty: str | None = None,
) -> list[CaseResult]:
    """一键评估入口。

    Usage:
        from eval import run_evaluation
        results = run_evaluation()                      # 全部用例
        results = run_evaluation(difficulty="hard")     # 仅困难用例
        results = run_evaluation(categories=["methodology"])
    """
    runner = EvalRunner()
    if dataset_path:
        runner.load_dataset(dataset_path)

    if categories or difficulty:
        results = runner.run_filtered(categories=categories, difficulty=difficulty)
    else:
        results = runner.run_all()

    # 生成报告
    report = runner.generate_report()
    print(report)

    # 持久化
    runner.save_results()

    return results


if __name__ == "__main__":
    run_evaluation()
