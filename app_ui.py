import langchain
langchain.debug = True
import asyncio
import threading
import queue
import chainlit as cl
from langchain_core.messages import HumanMessage

# 从后端导入大模型状态机
from agent_core import app 

@cl.on_chat_start
async def on_chat_start():
    """页面刷新时触发：拦截并询问用户 ID 以加载记忆"""
    res = await cl.AskUserMessage(
        content="🔑 **请输入您的会话/用户 ID (例如 u1001) 以加载历史记忆：**", 
        timeout=60
    ).send()
    
    if res and res.get('output'):
        thread_id = res['output'].strip()
    else:
        thread_id = "default_user"
        
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    
    welcome_msg = f"""
    ✅ **已成功挂载会话 [{thread_id}] 的持久化记忆！**
    
    # 🚀 全自动学术研究智能体
    您可以让我执行以下任务：
    * **全网检索：** 查找 Arxiv 或 Semantic Scholar 论文。
    * **自动入库：** 下载特定论文并消化到 Qdrant 向量库。
    * **深度研读：** 检索分析已有知识库中的论文细节。
    """
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """用户发送消息时触发：驱动后台 Agent 状态机"""

    config = cl.user_session.get("config")
    inputs = {"messages": [HumanMessage(content=message.content)]}

    res_msg = cl.Message(content="")
    current_step = None
    collected_sources = []  # 收集本轮所有工具的引用来源

    # =========================================================
    # 【核心架构升级】：多线程队列通信，彻底告别 WebSocket 超时断连
    # =========================================================

    # 1. 创建一个线程安全的通信队列
    event_queue = queue.Queue()

    # 2. 定义后台工人：在独立线程中运行沉重的 Agent 同步代码
    def run_agent_in_background():
        try:
            # 这里哪怕卡上 1 分钟，也不会影响主网页的连接！
            for event in app.stream(inputs, config=config, stream_mode="updates"):
                event_queue.put(("event", event))
            # 运行结束，放入结束信号
            event_queue.put(("done", None))
        except Exception as e:
            event_queue.put(("error", str(e)))

    # 3. 启动后台线程
    threading.Thread(target=run_agent_in_background, daemon=True).start()

    # 4. 主线程（网页UI）：通过异步方式非阻塞地监听队列
    while True:
        # 使用 asyncio.to_thread 优雅地挂起等待，保持 UI 心跳跳动
        msg_type, data = await asyncio.to_thread(event_queue.get)

        if msg_type == "done":
            break  # Agent 执行完毕，退出监听循环

        elif msg_type == "error":
            await cl.Message(content=f"❌ Agent 后台运行崩溃: {data}").send()
            break

        elif msg_type == "event":
            # 正常处理 Agent 返回的流式事件
            for node_name, node_state in data.items():

                # --- 场景 1：大模型思考与工具调用意图 ---
                if node_name == "agent":
                    last_message = node_state['messages'][-1]

                    if getattr(last_message, "tool_calls", None):
                        for tc in last_message.tool_calls:
                            current_step = cl.Step(name=tc['name'], type="tool")
                            current_step.input = tc['args']
                            await current_step.send()
                    elif last_message.content:
                        # 兜底：若 LLM 遗漏了来源，自动补上
                        content = last_message.content
                        if collected_sources and "📚" not in content:
                            unique_sources = list(dict.fromkeys(collected_sources))
                            content += "\n\n---\n📚 **引用来源**:\n" + "\n".join(unique_sources)
                        res_msg.content = content
                        await res_msg.send()

                # --- 场景 2：工具真正执行完毕 ---
                elif node_name == "tools":
                    if current_step:
                        last_message = node_state['messages'][-1]
                        # 提取工具输出中的引用来源行
                        tool_output = last_message.content
                        for line in tool_output.split("\n"):
                            if "📚" in line or "文献:" in line or "PDF链接:" in line or "引用来源" in line:
                                collected_sources.append(line.strip())
                        current_step.output = tool_output
                        current_step.status = "success"
                        await current_step.update()
                        current_step = None

                # --- 场景 3：分层记忆系统后台节点 ---
                elif node_name == "manage_context":
                    step = cl.Step(name="🧠 系统后台操作", type="run")
                    step.output = "检测到上下文达到阈值，已自动触发分层记忆整理与归档。"
                    step.status = "success"
                    await step.send()

                elif node_name == "cleanup_ephemeral":
                    # 静默：中间消息压缩为内部备忘
                    pass

                elif node_name == "retrieve_memory":
                    # 静默：长期记忆检索
                    pass

                elif node_name == "extract_memory":
                    # 静默：后台线程归档长期记忆
                    pass

                elif node_name == "update_profile":
                    # 静默：用户画像更新
                    pass

                elif node_name == "init_core":
                    # 静默：核心指令初始化
                    pass