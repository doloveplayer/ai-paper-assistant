"""线程安全的数据捕获器（Side-channel 模式）。

通过 thread-local singleton 在 eval runner 与被测管线之间传递数据。
不修改 LangChain 工具签名，正常使用时零开销。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class RetrievalTrace:
    """单次 search_vision_knowledge 调用的完整检索轨迹。"""
    query: str
    paper_id: str | None
    vision_hits: list[dict]       # ScoredPoint 序列化结果（视觉路）
    text_hits: list[dict]         # ScoredPoint 序列化结果（文本路）
    fused_hits: list[dict]        # RRF 融合后截断的结果
    final_pages_info: list[dict]  # 最终送给 VLM 的页面信息 [{doc_id, page_num, raw_text}]
    final_answer: str             # search_vision_knowledge 的完整返回字符串


@dataclass
class TelemetryTrace:
    """单次 call_model 调用的效能遥测数据。"""
    e2e_latency: float
    input_tokens: int
    output_tokens: int
    tpot: float          # Time Per Output Token (ms)
    throughput: float    # Tokens per second


@dataclass
class AgentStep:
    """Agent 执行的一个步骤（对应 LangGraph 的一个 node update）。"""
    node_name: str
    timestamp: float
    content_preview: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_output_preview: str = ""


class EvalContext:
    """线程局部的评估上下文。

    用法:
        ctx = EvalContext.enable()   # 开启捕获
        ...  # 运行 Agent
        traces = ctx.retrieval_traces
        EvalContext.disable()        # 关闭捕获
    """

    _local = threading.local()

    # ---------------------------------------------------------------
    # 类方法
    # ---------------------------------------------------------------
    @classmethod
    def get(cls) -> "EvalContext | None":
        return getattr(cls._local, "instance", None)

    @classmethod
    def enable(cls) -> "EvalContext":
        ctx = cls()
        cls._local.instance = ctx
        return ctx

    @classmethod
    def disable(cls):
        cls._local.instance = None

    # ---------------------------------------------------------------
    # 实例方法
    # ---------------------------------------------------------------
    def __init__(self):
        self.enabled = True
        self.retrieval_traces: list[RetrievalTrace] = []
        self.telemetry_traces: list[TelemetryTrace] = []
        self.agent_steps: list[AgentStep] = []
        self.final_answer: str = ""

    def capture_retrieval(
        self,
        query: str,
        paper_id: str | None,
        vision_hits: list[dict],
        text_hits: list[dict],
        fused_hits: list[dict],
    ):
        self.retrieval_traces.append(
            RetrievalTrace(
                query=query,
                paper_id=paper_id,
                vision_hits=vision_hits,
                text_hits=text_hits,
                fused_hits=fused_hits,
                final_pages_info=[],  # 后续由 runner 填充
                final_answer="",
            )
        )

    def capture_telemetry(
        self,
        e2e_latency: float,
        input_tokens: int,
        output_tokens: int,
        tpot: float,
        throughput: float,
    ):
        self.telemetry_traces.append(
            TelemetryTrace(
                e2e_latency=e2e_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tpot=tpot,
                throughput=throughput,
            )
        )

    def capture_agent_step(self, node_name: str, timestamp: float, **kwargs):
        self.agent_steps.append(AgentStep(node_name=node_name, timestamp=timestamp, **kwargs))

    # ---------------------------------------------------------------
    # 聚合属性（方便指标计算）
    # ---------------------------------------------------------------
    @property
    def total_latency(self) -> float:
        return sum(t.e2e_latency for t in self.telemetry_traces)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.telemetry_traces)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.telemetry_traces)

    @property
    def has_retrieval_data(self) -> bool:
        return len(self.retrieval_traces) > 0

    @property
    def last_retrieval(self) -> RetrievalTrace | None:
        return self.retrieval_traces[-1] if self.retrieval_traces else None
