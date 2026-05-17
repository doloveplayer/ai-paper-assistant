import os
import time
import base64
import re
import sqlite3
import json
import uuid
import threading
import datetime

from typing import TypedDict, Annotated, Sequence
from config import Config
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from tools.vision_rag import download_and_ingest_vision_paper, search_vision_knowledge, search_academic_papers
from tools.text_primary_rag import search_text_primary_knowledge
from tools.text_primary_ingest import ingest_paper_text_primary

# =========================================================
# 核心指令层 (Layer 1): 不可变 SOP — 供 vLLM Prefix Cache 长久缓存
# =========================================================
CORE_INSTRUCTIONS = (
    "You are an automated analytical API. Your ONLY mechanism for answering is executing tools.\n"
    "【执行 SOP】：\n"
    "0. 【新增】对于文本密集型查询（方法论细节、实验参数、公式讨论、数据集描述），优先使用 `search_text_primary_knowledge`。对架构图、图表数据提取等视觉密集型查询，使用 `search_vision_knowledge`。\n"
    "1. 首先使用工具`search_text_primary_knowledge`或`search_vision_knowledge`索引本地知识向量库内容，针对检索到的论文知识（图表/公式/架构）着重分析\n"
    "2. 且只要用户询问某篇**已有、刚才提到过、或已入库的论文细节**（包括任何图表、公式、实验数据、文字结论），**必须优先且直接调用 `search_vision_knowledge`**。\n"
    "3. 当本地知识库检索不到相关数据时候调用文献检索工具：`search_academic_papers`，一定使用英文关键词！！！！\n"
    "4. 【智能入库与连贯阅读闭环】（极其重要）：\n"
    "   - 当用户要求了解一篇**刚刚通过全网检索（search_academic_papers）发现、但还未入库的新论文**的详细内容时，你必须执行【两步走】策略：\n"
    "   - 第一步：先强行调用 `download_and_ingest_vision_paper` 将其下载并向量化。\n"
    "   - 第二步：等待入库成功后，你绝对不能停止思考！必须紧接着自动调用 `search_vision_knowledge` 去查询用户真正关心的问题细节，最后再把综合结果告诉用户。\n\n"
    "【CRITICAL RULES】:\n"
    "- DO NOT say '好的', '我将为您', '请稍等'.\n"
    "- DO NOT explain what you are going to do.\n"
    "- If you need to use a tool, call it via the function calling protocol — do NOT output the tool call JSON as plain text.\n"
    "- Only when you have received ALL tool results and are ready to answer the user, produce the final Markdown response.\n"
    "【CRITICAL RULES FOR FINAL RESPONSE】:\n"
    "当你已经获取了工具返回的信息，准备给用户进行最终总结时，你必须遵守以下规范：\n"
    "1. 绝对不要输出原始的 JSON、XML 或代码块结构！\n"
    "2. 必须使用极其优雅、易读的 Markdown 格式输出（使用二级标题、粗体、有序列表）。\n"
    "3. 保持专业学者的口吻，直接给出结论，不要复述你的思考过程。\n"
    "4. 【引用来源强制规范】你必须在回答末尾附上工具返回的「📚 **引用来源**」部分，明确标注每条信息的出处是「本地RAG向量库」还是「网页检索」。包括文献ID、页码、及可访问的PDF链接。严禁省略来源标注。"
)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """高效估算消息列表的总 Token 数"""
    total_tokens = 0
    for msg in messages:
        if getattr(msg, "tool_calls", None):
            total_tokens += 100
        content = str(msg.content)
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
            match = re.search(r'\[系统硬链接\]:\s*(\S+\.jpg)', msg.content)
            if match:
                img_path = match.group(1).strip()
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
# 1. 分层记忆状态定义
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    profile: dict               # Layer 2: 用户画像 {"research_interests": [...], "mentioned_papers": [...], "preferences": {...}}
    working_context: list[dict]  # Layer 3: 最近 Q&A [{"question": str, "answer": str, "timestamp": str}, ...]
    core_instructions: str       # Layer 1: 不可变 SOP

# ---------------------------------------------------------
# 2. 准备工具与大模型
# ---------------------------------------------------------
tools = [
    search_vision_knowledge,
    search_academic_papers,
    download_and_ingest_vision_paper,
    search_text_primary_knowledge,
    ingest_paper_text_primary,
]

llm = ChatOpenAI(
    base_url=Config.LLM_API_BASE,
    api_key="EMPTY",
    model=Config.LLM_MODEL_NAME
)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# 3. 图节点定义
# ---------------------------------------------------------

def init_core_instructions(state: AgentState) -> dict:
    """Layer 1: 在 START 后立即注入不可变核心指令"""
    return {"core_instructions": CORE_INSTRUCTIONS}


def call_model(state: AgentState):
    """
    Layer 1-3 融合 + 缓存友好 Prompt 拼接。
    关键设计：静态头部 → 半静态中部 → 动态尾部，确保 vLLM prefix cache 命中率 ≥60%。
    """
    messages = state["messages"]

    # ── 头部 (绝对静态): core_instructions — vLLM 长久缓存 ──
    core = state.get("core_instructions", CORE_INSTRUCTIONS)
    system_parts = [core]

    # ── 中部 (半静态): working_context — 只有尾部变动，前半部分命中缓存 ──
    working_context = state.get("working_context", [])
    if working_context:
        ctx_text = "\n\n【近期对话】:\n"
        for entry in working_context[-Config.WORKING_CONTEXT_MAX_ROUNDS:]:
            ctx_text += f"Q: {entry.get('question', '')[:200]}\n"
            ctx_text += f"A: {entry.get('answer', '')[:300]}\n\n"
        system_parts.append(ctx_text)

    # ── 尾部 (极度动态): profile + Qdrant 记忆事实 — 紧贴本轮用户 query ──
    profile = state.get("profile", {})
    if profile:
        interests = profile.get("research_interests", [])
        mentioned = profile.get("mentioned_papers", [])
        if interests or mentioned:
            profile_text = "\n\n【用户画像】:\n"
            if interests:
                profile_text += f"研究兴趣: {', '.join(interests)}\n"
            if mentioned:
                profile_text += f"曾关注论文: {', '.join(mentioned)}\n"
            system_parts.append(profile_text)

    sys_message = SystemMessage(content="\n".join(system_parts))
    invoke_messages = [sys_message] + messages

    # ⏱️ 性能遥测
    start_time = time.time()
    response = llm_with_tools.invoke(invoke_messages)
    end_time = time.time()

    e2e_latency = end_time - start_time
    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    tpot = (e2e_latency / output_tokens) if output_tokens > 0 else 0
    throughput = (output_tokens / e2e_latency) if e2e_latency > 0 else 0

    print(f"\n📊 [效能面板] E2E延时: {e2e_latency:.2f}s | "
          f"输入: {input_tokens} tokens | 输出: {output_tokens} tokens | "
          f"吞吐量: {throughput:.1f} t/s | TPOT: {tpot*1000:.1f} ms/t")

    # 评估系统追踪
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


def cleanup_ephemeral(state: AgentState) -> dict:
    """
    Layer 4: 临时执行层 — 替换 (非删除) 中间工具结果。
    将冗长的 ToolMessage 替换为轻量内部备忘，保留短期溯源能力，
    同时释放 ~90% 上下文 Token 占用。
    """
    messages = state["messages"]
    if not messages:
        return {}

    last_msg = messages[-1]
    # 仅在 agent 给出最终回答后触发（非工具调用中）
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return {}

    replacements = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, 'name', '')
            content = str(msg.content)

            if tool_name == 'search_vision_knowledge':
                # 从 RAG 结果中提取论文 ID 和页码
                paper_ids = re.findall(r'(?:paper[_\s]?id[:\s]*|[（(])(\d{4}\.\d{4,5})', content, re.I)
                pages = re.findall(r'第\s*(\d+)\s*页|page\s*(\d+)', content, re.I)
                paper_list = ', '.join(paper_ids[:3]) if paper_ids else '未知'
                page_list = ', '.join(set(p for t in pages for p in t if p)) if pages else '未知'

                memo = (
                    f"[内部备忘] 上一轮通过 search_vision_knowledge 检索到了文献 {paper_list}，"
                    f"涉及第 {page_list} 页。"
                    f"如需回溯详细数据（完整图表、公式、实验数据），请再次调用 search_vision_knowledge。"
                )

            elif tool_name == 'search_academic_papers':
                # 从搜索结果中提取论文 ID
                paper_ids = re.findall(r'\b(\d{4}\.\d{4,5})\b', content)
                paper_list = ', '.join(paper_ids[:5]) if paper_ids else '未知'
                memo = (
                    f"[内部备忘] 上一轮通过 search_academic_papers 搜索到了以下论文: {paper_list}。"
                    f"如需重新检索或查看详情，请再次调用相应工具。"
                )

            elif tool_name == 'download_and_ingest_vision_paper':
                paper_ids = re.findall(r'\b(\d{4}\.\d{4,5})\b', content)
                paper_list = ', '.join(paper_ids[:3]) if paper_ids else '未知'
                memo = (
                    f"[内部备忘] 上一轮已将文献 {paper_list} 下载并入库到本地向量数据库。"
                    f"现在可以通过 search_vision_knowledge 查询其详细内容。"
                )

            elif tool_name == 'search_text_primary_knowledge':
                paper_ids = re.findall(r'(?:paper[_\s]?id[:\s]*|[（(])(\d{4}\.\d{4,5})', content, re.I)
                pages = re.findall(r'第\s*(\d+)\s*页|page\s*(\d+)', content, re.I)
                paper_list = ', '.join(paper_ids[:3]) if paper_ids else '未知'
                page_list = ', '.join(set(p for t in pages for p in t if p)) if pages else '未知'
                memo = (
                    f"[内部备忘] 上一轮通过 search_text_primary_knowledge 检索到了文献 {paper_list}，"
                    f"涉及第 {page_list} 页。"
                    f"如需回溯详细数据，请再次调用 search_text_primary_knowledge。"
                )

            elif tool_name == 'ingest_paper_text_primary':
                paper_ids = re.findall(r'\b(\d{4}\.\d{4,5})\b', content)
                paper_list = ', '.join(paper_ids[:3]) if paper_ids else '未知'
                memo = (
                    f"[内部备忘] 上一轮已将文献 {paper_list} 下载并入库到文本主路向量数据库。"
                    f"现在可以通过 search_text_primary_knowledge 查询其详细内容。"
                )

            else:
                # 其他未知工具：保留首 200 字符作为摘要
                truncated = content[:200] + "..." if len(content) > 200 else content
                memo = f"[内部备忘] 工具 {tool_name} 返回结果 (已截断): {truncated}"

            replacements.append(RemoveMessage(id=msg.id))
            replacements.append(AIMessage(content=memo))

    # 同时清理包含 tool_calls 的 AIMessage（工具调用指令本身）
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            replacements.append(RemoveMessage(id=msg.id))

    if replacements:
        print(f"🧹 [内存清理] 将 {len(replacements)} 条中间消息替换为轻量备忘")
        return {"messages": replacements}
    return {}


def manage_working_context(state: AgentState) -> dict:
    """
    Layer 3: 管理工作上下文 — 保留最近 N 轮对话，旧消息移除。
    替代原有的 summarize_conversation。
    """
    messages = state["messages"]
    working_context = list(state.get("working_context", []))

    # 从 messages 中提取 HumanMessage + 最终 AIMessage (不含 tool_calls) 配对
    pairs = []
    i = 0
    while i < len(messages):
        if isinstance(messages[i], HumanMessage):
            question = messages[i].content
            answer = ""
            j = i + 1
            while j < len(messages):
                if isinstance(messages[j], AIMessage) and not getattr(messages[j], 'tool_calls', None):
                    answer = messages[j].content
                    break
                j += 1
            if answer:
                pairs.append({"question": question, "answer": answer, "timestamp": datetime.datetime.now().isoformat()})
            i = j + 1 if j < len(messages) else i + 1
        else:
            i += 1

    # 合并并截断到最大轮数
    all_pairs = working_context + pairs
    if len(all_pairs) > Config.WORKING_CONTEXT_MAX_ROUNDS:
        all_pairs = all_pairs[-Config.WORKING_CONTEXT_MAX_ROUNDS:]

    # 物理删除旧消息，保留最后 2 条 (AI 回答 + 用户新问题)
    delete_ops = []
    if len(messages) > 2:
        delete_ops = [RemoveMessage(id=m.id) for m in messages[:-2]]

    print(f"🧠 [上下文管理] 保留 {len(all_pairs)} 轮对话，移除 {len(delete_ops)} 条旧消息")
    return {
        "working_context": all_pairs,
        "messages": delete_ops
    }


def update_profile(state: AgentState) -> dict:
    """
    Layer 2: 实体与偏好层 — 从用户消息中提取论文 ID 和研究兴趣，更新动态画像。
    仅当检测到变化时更新 state.profile。
    """
    messages = state["messages"]
    profile = dict(state.get("profile", {}))

    recent_user_msgs = [m for m in messages if isinstance(m, HumanMessage)][-3:]
    recent_text = " ".join([str(m.content)[:500] for m in recent_user_msgs])

    changed = False

    # 提取论文 ID
    paper_ids = re.findall(r'\b(\d{4}\.\d{4,5})\b', recent_text)
    if paper_ids:
        mentioned = set(profile.get("mentioned_papers", []))
        new_ids = set(paper_ids) - mentioned
        if new_ids:
            mentioned.update(new_ids)
            profile["mentioned_papers"] = list(mentioned)
            changed = True

    # 研究兴趣关键词检测
    interest_patterns = {
        "object_detection": r'(object detection|目标检测|YOLO|DETR|detector)',
        "domain_adaptation": r'(domain adapt|域自适应|domain shift|domain gap|unsupervised domain)',
        "adverse_weather": r'(adverse weather|恶劣天气|fog|rain|low.?light|haze|nighttime)',
        "multimodal": r'(multi.?modal|多模态|vision.?language|VLM|visual question)',
        "image_restoration": r'(dehazing|deblurring|denoising|去雾|去噪|超分|super.?resolution)',
        "transformer": r'(transformer|attention|ViT|Swin|self.?attention)',
        "few_shot": r'(few.?shot|zero.?shot|小样本|零样本|domain generalization)',
    }

    existing_interests = set(profile.get("research_interests", []))
    for interest, pattern in interest_patterns.items():
        if re.search(pattern, recent_text, re.IGNORECASE):
            if interest not in existing_interests:
                existing_interests.add(interest)
                changed = True

    if existing_interests != set(profile.get("research_interests", [])):
        profile["research_interests"] = list(existing_interests)

    if changed:
        print(f"📝 [用户画像] 已更新: {profile}")
        return {"profile": profile}
    return {}


def _do_extract_and_store(messages_copy: list, profile_copy: dict, question: str, answer: str):
    """后台线程：提取记忆事实 → BGE-M3 编码 → Qdrant 存储"""
    try:
        from tools.vision_rag import _get_text_model

        # 调用 LLM 提取记忆事实
        extraction_prompt = (
            "请从以下对话中提取重要的记忆事实。每行一个事实，格式为：\n"
            "「论文关注:xxx」或「用户偏好:xxx」或「研究发现:xxx」\n\n"
            f"用户: {question[:500]}\n"
            f"助手: {answer[:500]}\n\n"
            "只输出有意义的记忆事实，每条不超过100字。如果无重要信息，输出「NONE」。"
        )

        import requests
        response = requests.post(
            f"{Config.LLM_API_BASE}/chat/completions",
            json={
                "model": Config.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": extraction_prompt}],
                "max_tokens": 512,
                "temperature": 0.1
            },
            timeout=20
        )

        if response.status_code != 200:
            return

        facts_text = response.json()['choices'][0]['message']['content']
        if 'NONE' in facts_text:
            return

        # BGE-M3 编码
        try:
            text_model = _get_text_model()
        except Exception as e:
            print(f"⚠️ 无法加载 BGE-M3 编码器: {e}")
            return

        # 创建 Qdrant 连接
        import qdrant_client
        from qdrant_client.models import PointStruct, VectorParams, Distance

        client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)

        # 确保 user_memory 集合存在
        if not client.collection_exists(Config.USER_MEMORY_COLLECTION):
            client.create_collection(
                collection_name=Config.USER_MEMORY_COLLECTION,
                vectors_config={
                    "dense": VectorParams(
                        size=Config.BEG_EMBEDDING_SIZE,
                        distance=Distance.COSINE
                    )
                }
            )

        timestamp = datetime.datetime.now().isoformat()
        points = []

        for fact_line in facts_text.strip().split('\n'):
            fact_line = fact_line.strip()
            if not fact_line or len(fact_line) < 5:
                continue
            try:
                fact_embedding = text_model.encode(fact_line, normalize_embeddings=True).tolist()
                point_id = uuid.uuid4().hex
                points.append(PointStruct(
                    id=point_id,
                    vector={"dense": fact_embedding},
                    payload={
                        "text": fact_line,
                        "timestamp": timestamp,
                        "type": "memory_fact"
                    }
                ))
            except Exception as e:
                print(f"⚠️ 编码记忆事实失败: {e}")

        if points:
            client.upsert(
                collection_name=Config.USER_MEMORY_COLLECTION,
                points=points
            )
            print(f"🧠 [长期记忆] 已归档 {len(points)} 条事实到 Qdrant user_memory")

    except Exception as e:
        print(f"⚠️ 长期记忆提取失败 (非阻塞): {e}")


def extract_memory_facts(state: AgentState) -> dict:
    """
    Layer 5: 外部化长期记忆 — fire-and-forget 模式。
    在后台线程中提取事实并存入 Qdrant，不阻塞主流程。
    """
    messages = state["messages"]
    if len(messages) < 2:
        return {}

    # 找最后一条用户消息和最终 AI 回答
    user_msg = None
    ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and user_msg is None:
            user_msg = msg
        if isinstance(msg, AIMessage) and ai_msg is None and not getattr(msg, 'tool_calls', None):
            ai_msg = msg
        if user_msg and ai_msg:
            break

    if not user_msg or not ai_msg:
        return {}

    profile_copy = dict(state.get("profile", {}))
    thread = threading.Thread(
        target=_do_extract_and_store,
        args=(list(messages), profile_copy, str(user_msg.content), str(ai_msg.content)),
        daemon=True
    )
    thread.start()
    return {}


# ---------------------------------------------------------
# 4. 条件边 / 路由
# ---------------------------------------------------------

def retrieve_long_term_memory(state: AgentState) -> dict:
    """
    从 Qdrant user_memory 检索相关长期记忆事实，以合成消息形式注入 messages。
    这样后续 call_model 可以在动态尾部看到这些事实。
    """
    messages = state['messages']
    if not messages:
        return {}

    last_user_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m
            break

    if not last_user_msg:
        return {}

    try:
        from tools.vision_rag import _get_text_model
        import qdrant_client

        text_model = _get_text_model()
        query_embedding = text_model.encode(
            str(last_user_msg.content), normalize_embeddings=True
        ).tolist()

        client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)
        if client.collection_exists(Config.USER_MEMORY_COLLECTION):
            results = client.query_points(
                collection_name=Config.USER_MEMORY_COLLECTION,
                query=query_embedding,
                using="dense",
                limit=Config.MEMORY_RETRIEVAL_TOP_K
            ).points

            if results:
                memory_lines = []
                for hit in results:
                    payload = hit.payload or {}
                    memory_lines.append(f"- {payload.get('text', '')}")
                memory_text = "【长期记忆 - 相关历史信息】:\n" + "\n".join(memory_lines)
                print(f"🧠 [长期记忆] 检索到 {len(results)} 条相关历史事实")
                return {"messages": [HumanMessage(content=memory_text)]}

    except Exception as e:
        print(f"⚠️ 长期记忆检索失败 (非阻塞): {e}")

    return {}


def token_check_router(state: AgentState) -> str:
    """纯路由：Token 超标 → 上下文管理，否则 → Agent"""
    current_tokens = estimate_tokens(state['messages'])
    if current_tokens > 5000:
        print(f"📊 [系统监控] 当前 Token ({current_tokens}) 已超警戒线，进入上下文管理。")
        return "manage_context"
    return "agent"


def should_continue(state: AgentState) -> str:
    """Agent 思考后的路由：继续调工具 or 进入清理管线"""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "cleanup"


# ---------------------------------------------------------
# 5. 工具节点
# ---------------------------------------------------------
tool_node = ToolNode(tools)


# ---------------------------------------------------------
# 6. 组装终极状态机图
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("init_core", init_core_instructions)
workflow.add_node("retrieve_memory", retrieve_long_term_memory)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("manage_context", manage_working_context)
workflow.add_node("cleanup_ephemeral", cleanup_ephemeral)
workflow.add_node("update_profile", update_profile)
workflow.add_node("extract_memory", extract_memory_facts)

# START → init_core → retrieve_memory → token_check → agent or manage_context
workflow.add_edge(START, "init_core")
workflow.add_edge("init_core", "retrieve_memory")
workflow.add_conditional_edges("retrieve_memory", token_check_router, {
    "manage_context": "manage_context",
    "agent": "agent"
})

# manage_context → agent
workflow.add_edge("manage_context", "agent")

# agent ⇄ tools (loop) or → cleanup (final answer)
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "cleanup": "cleanup_ephemeral"
})
workflow.add_edge("tools", "agent")

# Post-answer pipeline: cleanup → update_profile → extract_memory → END
workflow.add_edge("cleanup_ephemeral", "update_profile")
workflow.add_edge("update_profile", "extract_memory")
workflow.add_edge("extract_memory", END)


# 统一使用全局数据库文件，靠 thread_id 区分用户
os.makedirs("chat_history", exist_ok=True)
db_path = os.path.join("chat_history", "global_agent_memory.sqlite")

conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------
# 7. 交互式终端循环 (CLI Loop)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 具备分层记忆与 RAG 检索能力的超级 Agent 已启动！(CLI 模式)")
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
            break

        if not user_input.strip():
            continue

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            for event in app.stream(inputs, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    if node_name == "agent":
                        # 只打印不含 tool_calls 的内容
                        last_msg = node_state['messages'][-1]
                        if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                            print(f"\n🤖 Agent:\n {last_msg.content}")
                    elif node_name == "tools":
                        print(f"\n🔧 [后台工具] 执行完毕...")
                    elif node_name == "manage_context":
                        print(f"\n🧠 [后台系统] 上下文已整理归档到分层记忆系统。")
                    elif node_name == "cleanup_ephemeral":
                        print(f"\n🧹 [后台系统] 中间过程消息已压缩为内部备忘。")
                    elif node_name == "retrieve_memory":
                        pass  # 长期记忆检索静默完成
                    elif node_name == "extract_memory":
                        print(f"\n📥 [后台系统] 长期记忆归档已触发 (后台线程)。")
                    elif node_name == "update_profile":
                        pass  # 静默
                    elif node_name == "init_core":
                        pass  # 静默
        except Exception as e:
            print(f"\n❌ 运行中断: {str(e)}")
