import os
import time
import base64
import re
import sqlite3

import json
import uuid
from langchain_core.messages import AIMessage

from typing import TypedDict, Annotated, Sequence
from config import Config
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

# 引入你的工具
# from tools.arxiv_tool import search_arxiv_metadata, download_and_ingest_paper,search_semantic_scholar, download_and_ingest_from_url
# from tools.query_qdrant import query_qdrant_knowledge
from tools.vision_rag import download_and_ingest_vision_paper, search_vision_knowledge,search_academic_papers


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """高效估算消息列表的总 Token 数"""
    total_tokens = 0
    for msg in messages:
        # 如果是工具调用指令，额外增加一些 Token 冗余
        if getattr(msg, "tool_calls", None):
            total_tokens += 100 
            
        content = str(msg.content)
        # 简单粗暴的工业级估算法：
        # 假设中英文混合，我们将总字符数除以 1.5 作为安全估算值
        # 你也可以把它设为 1，这样估算得更保守，绝对不会爆
        total_tokens += int(len(content) / 1.5) 
        
    return total_tokens

def encode_image_to_base64(image_path: str) -> str:
    """将本地图片读取为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def inject_vision_into_messages(messages: list) -> list:
    processed_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # 更灵活的正则匹配，允许 [系统硬链接]: 后有空白字符
            match = re.search(r'\[系统硬链接\]:\s*(\S+\.jpg)', msg.content)
            if match:
                img_path = match.group(1).strip()
                # 检查文件是否存在
                if os.path.exists(img_path):
                    try:
                        base64_image = encode_image_to_base64(img_path)
                        new_content = [
                            {"type": "text", "text": msg.content},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                        processed_messages.append(ToolMessage(content=new_content, tool_call_id=msg.tool_call_id, name=msg.name))
                        continue
                    except Exception as e:
                        print(f"⚠️ 无法读取本地图片 {img_path}: {e}")
                else:
                    print(f"⚠️ 图片文件不存在: {img_path}")
        processed_messages.append(msg)
    return processed_messages

# ---------------------------------------------------------
# 1. 定义全局状态 (带长期记忆和压缩机制)
# ---------------------------------------------------------
class AgentState(TypedDict):
    # 使用 add_messages 替代 operator.add，它允许我们通过传入 RemoveMessage 来删除旧历史
    messages: Annotated[list[BaseMessage], add_messages]
    # 新增 summary 字段，用于保存超出会话长度后的“浓缩记忆”
    summary: str

# ---------------------------------------------------------
# 2. 准备工具与大模型
# ---------------------------------------------------------
# tools = [
#     search_arxiv_metadata, 
#     search_semantic_scholar, 
#     download_and_ingest_paper, 
#     download_and_ingest_from_url, 
#     query_qdrant_knowledge,
# ]

tools = [
    search_vision_knowledge, 
    search_academic_papers, 
    download_and_ingest_vision_paper,
]

llm = ChatOpenAI(
    base_url=Config.LLM_API_BASE,
    api_key="EMPTY",
    model=Config.LLM_MODEL_NAME
)
# llm = ChatOpenAI(
#     model=Config.VLLM_MODEL_NAME,
#     base_url=Config.LLM_API_BASE,
#     api_key="EMPTY",
# )
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState):
    """节点 1：大脑思考与效能遥测"""
    summary = state.get("summary", "")
    messages = state["messages"]

    system_prompt = (
        "You are an automated analytical API. Your ONLY mechanism for answering is executing tools.\n"
        "【执行 SOP】：\n"
        "1. 首先使用工具`search_vision_knowledge`索引本地知识向量库只是内容，针对检索到的论文知识（图表/公式/架构）着重分析\n" \
        "2. 且只要用户询问某篇**已有、刚才提到过、或已入库的论文细节**（包括任何图表、公式、实验数据、文字结论），**必须优先且直接调用 `search_vision_knowledge`**。\n"
        "3. 当本地知识库检索不到相关数据时候调用文献检索工具：`search_academic_papers`，一定使用英文关键词！！！！\n"
        "4. 【智能入库与连贯阅读闭环】（极其重要）：\n"
        "   - 当用户要求了解一篇**刚刚通过全网检索（search_academic_papers）发现、但还未入库的新论文**的详细内容时，你必须执行【两步走】策略：\n"
        "   - 第一步：先强行调用 `download_and_ingest_vision_paper` 将其下载并向量化。\n"
        "   - 第二步：等待入库成功后，你绝对不能停止思考！必须紧接着自动调用 `search_vision_knowledge` 去查询用户真正关心的问题细节，最后再把综合结果告诉用户。\n\n"
        "【CRITICAL RULES】:\n"
        "- DO NOT say '好的', '我将为您', '请稍等'.\n"
        "- DO NOT explain what you are going to do.\n"
        "- Return ONLY the precise tool call arguments without any prefix or conversational text."
        "【CRITICAL RULES FOR FINAL RESPONSE】:\n"
        "当你已经获取了工具返回的信息，准备给用户进行最终总结时，你必须遵守以下规范：\n"
        "1. 绝对不要输出原始的 JSON、XML 或代码块结构！\n"
        "2. 必须使用极其优雅、易读的 Markdown 格式输出（使用二级标题、粗体、有序列表）。\n"
        "3. 保持专业学者的口吻，直接给出结论，不要复述你的思考过程。\n"
        "4. 【引用来源强制规范】你必须在回答末尾附上工具返回的「📚 **引用来源**」部分，明确标注每条信息的出处是「本地RAG向量库」还是「网页检索」。包括文献ID、页码、及可访问的PDF链接。严禁省略来源标注。"
    )
    
    if summary:
        system_prompt += f"\n\n【历史对话浓缩记忆】:\n{summary}"
    
    sys_message = SystemMessage(content=system_prompt)
    invoke_messages = [sys_message] + messages
    # vision_ready_messages = inject_vision_into_messages(invoke_messages)
        
    # ==========================================
    # ⏱️ 性能遥测 (Telemetry) 拦截器
    # ==========================================
    start_time = time.time()
    
    # 真正发起 vLLM 请求
    response = llm_with_tools.invoke(invoke_messages)
    
    end_time = time.time()
    
    # 计算各项效能指标
    e2e_latency = end_time - start_time
    
    # 从 OpenAI 兼容的响应中提取 Token 信息 (LangChain 自动封装在 usage_metadata 中)
    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    # TPOT (Time Per Output Token): 生成每个 Token 的平均耗时
    # 注意：如果仅调用工具，output_tokens 可能很少
    tpot = (e2e_latency / output_tokens) if output_tokens > 0 else 0
    # Throughput (吞吐量): 每秒生成的 Token 数
    throughput = (output_tokens / e2e_latency) if e2e_latency > 0 else 0
    
    print(f"\n📊 [效能面板] E2E延时: {e2e_latency:.2f}s | "
          f"输入: {input_tokens} tokens | 输出: {output_tokens} tokens | "
          f"吞吐量: {throughput:.1f} t/s | TPOT: {tpot*1000:.1f} ms/t")

    # ---- 评估系统追踪点：捕获每次 LLM 调用的遥测数据 ----
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

def summarize_conversation(state: AgentState):
    """节点 3：记忆压缩器 (当上下文太长时触发)"""
    print("🧠 [系统] 记忆碎片过多，Agent 正在触发短期记忆压缩...")
    
    summary = state.get("summary", "")
    messages = state["messages"]

    # 【核心细节】：此时 messages 的最后一个元素是用户刚刚提出的新问题。
    # 我们绝对不能把新问题给总结并删除了！所以我们保留最后 2 条消息（上一轮的AI回答 + 本轮的用户新提问）
    messages_to_summarize = messages[:-2] if len(messages) > 2 else messages
    
    # 构建总结任务：让大模型把当前总结和旧对话融合
    summary_prompt = (
        f"请用简练的语言总结当前对话的核心信息，特别注意：必须保留用户明确否定的内容、以及重要的技术细节。如果你之前有总结，请将其结合。\n"
        f"之前的总结: {summary}\n"
        f"需要追加总结的对话: {messages_to_summarize}"
    )
    
    response = llm.invoke(summary_prompt)
    
    # 物理删除被总结的旧消息
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "summary": response.content, 
        "messages": delete_messages
    }

tool_node = ToolNode(tools)

# ---------------------------------------------------------
# 4. 定义条件边 (Conditional Edges)
# ---------------------------------------------------------

# 入口检测路由
def check_memory_before_agent(state: AgentState) -> str:
    """在 Agent 大脑开始思考前，先测量上下文总 Token"""
    messages = state['messages']
    current_tokens = estimate_tokens(messages)
    
    if current_tokens > 5000:
        print(f"📊 [系统监控] 当前 Token ({current_tokens}) 已超警戒线，先进入压缩节点。")
        return "summarize"
        
    return "agent"

def should_continue(state: AgentState) -> str:
    """路由逻辑：大模型思考完后，接下来走哪条路？"""
    messages = state['messages']
    last_message = messages[-1]
    
    # 1. 优先判断是否需要调用工具
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    # if last_message.tool_calls:
    #     return "continue"
        
    # 2. 否则，直接结束本轮回答
    return "end"

# ---------------------------------------------------------
# 5. 组装终极状态机图 (The Graph)
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("summarize_conversation", summarize_conversation) # 注册总结节点

# 1. 图的入口不再直达 Agent，而是先过一道安检 (check_memory_before_agent)
workflow.add_conditional_edges(START, check_memory_before_agent, {
    "summarize": "summarize_conversation", 
    "agent": "agent"
})

# 2. 如果去执行了记忆压缩，压缩完毕后，必须无条件前往 Agent 去回答用户刚提出的问题！
workflow.add_edge("summarize_conversation", "agent")

# 3. Agent 执行完毕后的常规模型反馈分流
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools", 
    "end": END
})

workflow.add_edge("tools", "agent")


# 【核心重构】：统一使用一个全局的数据库文件，靠 thread_id 区分用户
os.makedirs("chat_history", exist_ok=True)
db_path = os.path.join("chat_history", "global_agent_memory.sqlite")

# 在全局范围连接数据库并编译图，这样外部模块就能直接 import app 了！
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

# ---------------------------------------------------------
# 6. 交互式终端循环 (CLI Loop)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 具备长期记忆与 RAG 检索能力的超级 Agent 已启动！(CLI 模式)")
    print("="*50)
    
    thread_id = input("🔑 请输入您的会话/用户 ID (例如 u1001): ").strip()
    if not thread_id:
        thread_id = "default_user"
        
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"✅ 已加载全局数据库。当前隔离会话: [{thread_id}]。您可以开始对话了。(输入 'quit' 退出)\n")
    
    while True:
        user_input = input("\n🧑 你:\n ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Agent 已休眠，记忆已安全保存。")
            # 注意：如果单纯跑脚本，退出时最好断开连接，但不影响 Web 端
            break
            
        if not user_input.strip():
            continue

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        try:
            for event in app.stream(inputs, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    if node_name == "agent":
                        print(f"\n🤖 Agent:\n {node_state['messages'][-1].content}")
                    elif node_name == "tools":
                        print(f"\n🔧 [后台工具] 执行完毕...")
                    elif node_name == "summarize_conversation":
                        print(f"\n🧠 [后台系统] 历史记忆已成功压缩。")
        except Exception as e:
            print(f"\n❌ 运行中断: {str(e)}")