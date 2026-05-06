#!/usr/bin/env python3
"""
extract_to_worksheet.py — 半自动数据蒸馏：挖矿与打包
=====================================================

从 Qdrant 向量数据库和 SQLite 对话历史中提取原始训练数据，打包为
人工可编辑的 "Worksheet" 文件。流程分三路：

  Summarizer 路: Qdrant page_text → 5 个 .txt batch 文件 (每批 10 条)
  Vision 路:     Qdrant page_tables + 图片 → 复制图片 + Markdown 作业表
  Planner 路:    SQLite checkpoint → 直接生成 planner_data.jsonl (全自动)

运行方式:
  conda run -n ai_agent python scripts/extract_to_worksheet.py

输出目录结构:
  dataset/
  ├── tasks/
  │   ├── summarizer_batch_01.txt   # 每批含 System Prompt + 10 条带噪文本
  │   ├── summarizer_batch_02.txt
  │   ├── ...
  │   ├── vision_worksheet.md       # 图片编号 + 配套表格，供大模型分析
  │   └── vision_images/
  │       ├── vis_01.jpg            # 从缓存复制的页面图片
  │       ├── vis_02.jpg
  │       └── ...
  ├── planner_data.jsonl            # 直接可用的 LLaMA-Factory 格式
  └── answers/                      # (空目录，等待人工填入大模型回答)
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import uuid
import pickle
import sqlite3
import random
import re
from typing import Optional

import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from tqdm import tqdm

from config import Config

# =========================================================
# 路径与常量
# =========================================================
DATASET_DIR = PROJECT_ROOT / "dataset"
TASKS_DIR = DATASET_DIR / "tasks"
VISION_IMG_DIR = TASKS_DIR / "vision_images"
ANSWERS_DIR = DATASET_DIR / "answers"

BATCH_SIZE = 10
SUMMARIZER_COUNT = 50
VISION_COUNT = 30

SYSTEM_PROMPT_SUMMARIZER = (
    "You are a professional academic text summarizer. Below are {count} noisy text snippets "
    "extracted from PDF papers.\n\n"
    "For each snippet, produce a concise academic summary in Chinese (≤200 words). "
    "Keep ALL specific model names (e.g. YOLOv8, DETR, ResNet-50), numerical metrics "
    "(mAP, AP50, F1, etc.), and key experimental findings. Use Markdown formatting.\n\n"
    "CRITICAL: Return ONLY a valid JSON array. Each element MUST have exactly two keys:\n"
    '  - "id": the exact data ID string as given (do not modify)\n'
    '  - "summary": your refined summary string (Markdown, ≤200 words)\n\n'
    'Example output:\n'
    '[{{"id": "2403.02148_page5", "summary": "YOLOv8 在雾天场景下..."}},\n'
    ' {{"id": "2310.xxxxx_page3", "summary": "..."}}]\n\n'
    "DO NOT add any text before or after the JSON array."
)

SYSTEM_PROMPT_VISION = (
    "You are a vision-language academic analyst. Given a paper page image and its "
    "extracted table/chart data, provide a detailed analysis.\n\n"
    "For each image-table pair below, write a thorough academic analysis in Chinese "
    "(≤300 words per image). Your analysis should:\n"
    "1. Identify the type of chart/table and what it compares\n"
    "2. Highlight key numerical results and trends\n"
    "3. Explain the significance of the findings in the paper's context\n\n"
    "CRITICAL: Return ONLY a valid JSON array. Each element MUST have exactly two keys:\n"
    '  - "image_id": the image ID (e.g. "vis_01")\n'
    '  - "analysis": your detailed analysis string (Markdown, ≤300 words)\n\n'
    'Example output:\n'
    '[{{"image_id": "vis_01", "analysis": "该表格展示了..."}},\n'
    ' {{"image_id": "vis_02", "analysis": "..."}}]\n\n'
    "DO NOT add any text before or after the JSON array."
)


def ensure_dirs():
    """确保所有输出目录存在"""
    for d in [DATASET_DIR, TASKS_DIR, VISION_IMG_DIR, ANSWERS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Summarizer 数据提取
# =========================================================

def extract_summarizer_worksheets():
    """
    从 Qdrant vrag_hybrid_collection 中 scroll 取 page_text 非空的 points，
    取前 SUMMARIZER_COUNT 条，分成 BATCH_SIZE 条一批，每批导出为一个 .txt 文件。
    """
    print("\n" + "=" * 60)
    print("📝 [Summarizer] 从 Qdrant 抽取文本摘要训练数据")
    print("=" * 60)

    client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)

    # Scroll 所有 points
    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=Config.COLLECTION_MIX_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    print(f"   共扫描 {len(all_points)} 个 points")

    # 过滤：page_text 非空且足够长 (>100 chars)
    valid = []
    for pt in all_points:
        payload = pt.payload or {}
        text = (payload.get("page_text") or "").strip()
        if len(text) > 100:
            valid.append({
                "data_id": f"{payload.get('paper_id', payload.get('file_name', 'unknown'))}_page{payload.get('page_number', '?')}",
                "page_text": text,
            })

    print(f"   有效 (page_text > 100 chars): {len(valid)} 条")

    # 随机采样
    if len(valid) > SUMMARIZER_COUNT:
        random.shuffle(valid)
        valid = valid[:SUMMARIZER_COUNT]

    if len(valid) == 0:
        print("   ⚠️ 无有效数据，跳过 Summarizer 提取")
        return

    # 分批写入 .txt
    batches = [valid[i:i + BATCH_SIZE] for i in range(0, len(valid), BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches):
        batch_file = TASKS_DIR / f"summarizer_batch_{batch_idx + 1:02d}.txt"
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT_SUMMARIZER.format(count=len(batch)))
            f.write("\n\n")
            f.write("=" * 60)
            f.write("\n\n")
            for item in batch:
                f.write(f'[ID: {item["data_id"]}]\n')
                f.write(f'{item["page_text"]}\n')
                f.write("\n---\n\n")

        print(f"   ✅ batch_{batch_idx + 1:02d}: {len(batch)} 条 → {batch_file.name}")

    print(f"   📦 共 {len(valid)} 条，{len(batches)} 个 batch 文件")


# =========================================================
# 2. Vision 数据提取
# =========================================================

def extract_vision_worksheets():
    """
    从 Qdrant 中抽取 page_tables 非空且 image_path 有效的 points，
    复制图片到 vision_images/ 并重命名，导出 Markdown 作业表。
    """
    print("\n" + "=" * 60)
    print("👁️  [Vision] 从 Qdrant 抽取视觉分析训练数据")
    print("=" * 60)

    client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)

    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=Config.COLLECTION_MIX_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    # 过滤：page_tables 非空 且 image_path 存在
    valid = []
    for pt in all_points:
        payload = pt.payload or {}
        tables = payload.get("page_tables") or {}
        img_path = (payload.get("image_path") or "").strip()
        if tables and img_path and os.path.exists(img_path):
            valid.append({
                "data_id": f"{payload.get('paper_id', payload.get('file_name', 'unknown'))}_page{payload.get('page_number', '?')}",
                "paper_id": payload.get("paper_id", payload.get("file_name", "unknown")),
                "page_number": payload.get("page_number", "?"),
                "image_path": img_path,
                "page_tables": tables,
                "page_text": (payload.get("page_text") or "")[:500],
            })

    print(f"   有效 (含表格且图片存在): {len(valid)} 条")

    if len(valid) > VISION_COUNT:
        random.shuffle(valid)
        valid = valid[:VISION_COUNT]

    if len(valid) == 0:
        print("   ⚠️ 无有效数据，跳过 Vision 提取")
        return

    # 复制图片并构建 Markdown 作业表
    md_lines = [
        "# Vision Analysis Worksheet",
        "",
        "请对以下每张图片进行详细的学术图表分析（中文，≤300 字）。",
        "分析时请结合提供的 Markdown 表格数据和页面上下文文本。",
        "",
        "## 分析要求",
        "1. 识别图表类型及其对比内容",
        "2. 提取关键数值结果和趋势",
        "3. 结合论文上下文解释研究发现的意义",
        "",
        "## 输出格式",
        "请将全部 {count} 张图片的分析结果保存为如下 JSON 文件，",
        "命名为 `vision_answers.json` 放入 `dataset/answers/` 目录：",
        "",
        '```json',
        '[',
        '  {"image_id": "vis_01", "analysis": "该表格展示了..."},',
        '  {"image_id": "vis_02", "analysis": "..."},',
        '  ...',
        ']',
        '```',
        "",
        "---",
        "",
    ]

    metadata_records = []  # 保存元数据供 build 脚本使用

    for idx, item in enumerate(valid):
        vis_id = f"vis_{idx + 1:02d}"
        src_path = item["image_path"]
        dst_filename = f"{vis_id}.jpg"
        dst_path = VISION_IMG_DIR / dst_filename

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"   ⚠️ 复制图片失败 {vis_id}: {e}")
            continue

        # 格式化表格为 Markdown 字符串
        tables_md = ""
        for t_id, md_content in item["page_tables"].items():
            tables_md += f"\n**{t_id}**:\n\n{md_content}\n"

        md_lines.append(f"## Image {vis_id} (`{dst_filename}`)")
        md_lines.append("")
        md_lines.append(f"- **Paper ID**: `{item['paper_id']}`")
        md_lines.append(f"- **Page**: {item['page_number']}")
        md_lines.append(f"- **Context Text** (前 500 字符): {item['page_text'][:500]}")
        md_lines.append("")
        md_lines.append("### 配套表格/图表数据 (Markdown)")
        md_lines.append("")
        md_lines.append(tables_md if tables_md else "(无表格数据)")
        md_lines.append("")
        md_lines.append("### 你的分析 (请填写)")
        md_lines.append("")
        md_lines.append("> _(在此处填写分析结果)_")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        metadata_records.append({
            "image_id": vis_id,
            "image_filename": dst_filename,
            "image_path": str(dst_path.resolve()),
            "paper_id": item["paper_id"],
            "page_number": item["page_number"],
            "page_tables": item["page_tables"],
            "data_id": item["data_id"],
        })

    # 写入 Markdown 作业表
    md_file = TASKS_DIR / "vision_worksheet.md"
    md_content = "\n".join(md_lines).replace("{count}", str(len(metadata_records)))
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 保存元数据 JSON 供 build 脚本使用
    metadata_file = TASKS_DIR / "vision_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 复制 {len(metadata_records)} 张图片 → {VISION_IMG_DIR}/")
    print(f"   ✅ 作业表 → {md_file.name}")
    print(f"   ✅ 元数据 → {metadata_file.name}")


# =========================================================
# 3. Planner 数据提取 (全自动，直接从 SQLite 生成 JSONL)
# =========================================================

def extract_planner_data():
    """
    从 LangGraph SQLite checkpoint 中提取带 tool_calls 的对话，
    直接生成 LLaMA-Factory 格式的 planner_data.jsonl。

    策略：
    - 从 SQLite 直接读取所有 thread_id
    - 用 thread_id 过滤非评估会话 (排除 eval_ 前缀)
    - 每个 thread_id 取最新 checkpoint，获取完整对话历史
    - 从完整对话中提取 (HumanMessage → AIMessage-with-tool_calls) 对

    格式: {"messages": [
        {"role": "system", "content": "SOP..."},
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "[{\"name\": \"...\", \"args\": {...}}]"}
    ]}
    """
    print("\n" + "=" * 60)
    print("🧠 [Planner] 从 SQLite checkpoint 提取工具调用对话")
    print("=" * 60)

    db_path = PROJECT_ROOT / "chat_history" / "global_agent_memory.sqlite"
    if not db_path.exists():
        print(f"   ⚠️ 数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        saver = SqliteSaver(conn)
        samples = _extract_via_langgraph(saver, conn)
    except Exception as e:
        print(f"   ⚠️ langgraph API 异常 ({e})，尝试 raw 回退")
        import traceback
        traceback.print_exc()
        samples = _extract_via_raw_sqlite(conn)

    conn.close()

    if not samples:
        print("   ⚠️ 未提取到有效的 planner 训练样本")
        return

    # 去重
    seen = set()
    unique = []
    for s in samples:
        h = hash(s["messages"][1]["content"][:100] + s["messages"][2]["content"][:100])
        if h not in seen:
            seen.add(h)
            unique.append(s)

    # 写入 JSONL
    output_file = DATASET_DIR / "planner_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in unique:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   ✅ 提取 {len(unique)} 条 (去重后) → {output_file.name}")


def _extract_via_langgraph(saver, conn) -> list:
    """
    通过 LangGraph SqliteSaver API 提取对话。

    关键修复点:
    1. 先从 SQLite 获取所有 thread_id，再按 thread_id 过滤 saver.list()
    2. 每个 thread 取最新 checkpoint (包含完整对话历史)
    3. 消息在 cp['channel_values']['messages'] 中，不是 cp['messages']
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # 从 SQLite 获取所有 thread_id
    rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    all_thread_ids = [r[0] for r in rows]

    # 过滤：排除 eval_ 前缀的评估会话，只保留真实对话
    real_threads = [tid for tid in all_thread_ids if not tid.startswith("eval_")]
    print(f"   共 {len(all_thread_ids)} 个 thread，其中 {len(real_threads)} 个真实对话")

    if not real_threads:
        print("   ⚠️ 无真实对话 (所有 thread 以 eval_ 开头)，尝试包含评估会话")
        real_threads = all_thread_ids

    samples = []

    for tid in tqdm(real_threads, desc="   处理 thread"):
        try:
            # 获取该 thread 的所有 checkpoint (按时间排序)
            thread_configs = list(saver.list(
                {"configurable": {"thread_id": tid}}
            ))

            if not thread_configs:
                continue

            # 取最新 checkpoint: saver.list() 按 step 降序排列 (最新最先)
            # 但为安全起见，显式按 metadata.step 排序取最大值
            thread_configs.sort(key=lambda t: t.metadata.get("step", -1) if t.metadata else -1, reverse=True)
            latest_tuple = thread_configs[0]
            state = saver.get_tuple(latest_tuple.config)

            if state is None or state.checkpoint is None:
                continue

            cp = state.checkpoint
            # 消息在 channel_values 内部
            messages = cp.get("channel_values", {}).get("messages", [])
            if not messages:
                continue

            # 从完整对话中提取 (HumanMessage, AIMessage-with-tool_calls) 对
            for i in range(len(messages) - 1):
                msg = messages[i]
                next_msg = messages[i + 1]

                if not isinstance(msg, HumanMessage):
                    continue
                if not isinstance(next_msg, AIMessage):
                    continue
                if not hasattr(next_msg, "tool_calls") or not next_msg.tool_calls:
                    continue

                # 过滤：只保留有效 tool_calls (非空 name)
                tool_calls_json = []
                for tc in next_msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    if name:
                        tool_calls_json.append({"name": name, "args": args})

                if not tool_calls_json:
                    continue

                samples.append({
                    "messages": [
                        {"role": "system", "content": _get_planner_system_prompt()},
                        {"role": "user", "content": str(msg.content)},
                        {"role": "assistant", "content": json.dumps(tool_calls_json, ensure_ascii=False)},
                    ]
                })

        except Exception as e:
            print(f"   ⚠️ thread {tid} 处理异常: {e}")
            continue

    return samples


def _extract_via_raw_sqlite(conn) -> list:
    """
    通过 msgpack 反序列化 checkpoint BLOB (回退方案)。

    LangGraph v0.2+ 使用 msgpack 序列化 checkpoint，
    数据在 checkpoint['channel_values']['messages'] 中。
    """
    try:
        import msgpack
    except ImportError:
        print("   ⚠️ msgpack 未安装，尝试 pip install msgpack")
        return []

    samples = []

    # 取每个 thread 的最新 checkpoint_id
    rows = conn.execute("""
        SELECT c.thread_id, c.checkpoint
        FROM checkpoints c
        INNER JOIN (
            SELECT thread_id, MAX(checkpoint_id) as max_id
            FROM checkpoints
            GROUP BY thread_id
        ) latest ON c.thread_id = latest.thread_id AND c.checkpoint_id = latest.max_id
        WHERE c.thread_id NOT LIKE 'eval_%'
    """).fetchall()

    if not rows:
        print("   ⚠️ 无真实对话数据")
        return []

    for thread_id, checkpoint_blob in tqdm(rows, desc="   解析 checkpoint"):
        try:
            cp = msgpack.unpackb(checkpoint_blob, raw=False)
        except Exception:
            continue

        # 消息在 channel_values 内
        channel_values = cp.get("channel_values", {})
        messages = channel_values.get("messages", [])
        if not messages:
            continue

        # msgpack 还原的消息为 dict 列表形式
        for i in range(len(messages) - 1):
            msg = messages[i]
            next_msg = messages[i + 1]

            # msgpack 还原: langgraph 使用特殊类型编码
            # HumanMessage -> {"type": "human", "content": ...}
            # AIMessage -> {"type": "ai", "content": ..., "tool_calls": [...]}
            if not isinstance(msg, dict) or not isinstance(next_msg, dict):
                continue

            if msg.get("type") != "human":
                continue
            if next_msg.get("type") != "ai":
                continue

            tool_calls = next_msg.get("tool_calls")
            if not tool_calls:
                continue

            samples.append({
                "messages": [
                    {"role": "system", "content": _get_planner_system_prompt()},
                    {"role": "user", "content": str(msg.get("content", ""))},
                    {"role": "assistant", "content": json.dumps(tool_calls, ensure_ascii=False)},
                ]
            })

    return samples


def _get_planner_system_prompt() -> str:
    """Planner LoRA 微调用的 System Prompt (精简版 SOP)"""
    return (
        "You are an academic research agent. Your task is to decide which tool to call "
        "based on the user's question.\n\n"
        "Available tools:\n"
        "1. search_vision_knowledge(query, paper_id?) — Search local paper database\n"
        "2. search_academic_papers(primary_concept, secondary_concept, max_results) — Web search\n"
        "3. download_and_ingest_vision_paper(paper_id, pdf_url) — Download and ingest paper\n\n"
        "Rules:\n"
        "- For paper details → use search_vision_knowledge first\n"
        "- If not found locally → use search_academic_papers\n"
        "- For new un-ingested papers → download_and_ingest_vision_paper then search_vision_knowledge\n"
        "- Return ONLY a JSON array of tool calls, no extra text."
    )


# =========================================================
# 主入口
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 VRAG 微调数据抽取工具")
    print("=" * 60)

    ensure_dirs()

    try:
        extract_summarizer_worksheets()
    except Exception as e:
        print(f"❌ Summarizer 提取失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        extract_vision_worksheets()
    except Exception as e:
        print(f"❌ Vision 提取失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        extract_planner_data()
    except Exception as e:
        print(f"❌ Planner 提取失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 数据提取完成！")
    print(f"   输出目录: {DATASET_DIR}/")
    print(f"   Summarizer 作业: {TASKS_DIR}/summarizer_batch_*.txt")
    print(f"   Vision 作业:     {TASKS_DIR}/vision_worksheet.md")
    print(f"   Vision 图片:     {VISION_IMG_DIR}/")
    print(f"   Planner JSONL:   {DATASET_DIR}/planner_data.jsonl")
    print()
    print("📋 下一步: 将作业文件交给大模型 (GPT-4o / DeepSeek) 生成回答，")
    print("   把回答保存为 JSON 放到 dataset/answers/ 目录，")
    print("   然后运行: python scripts/build_llama_factory_jsonl.py")
    print("=" * 60)
