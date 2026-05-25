#!/usr/bin/env python3
"""
build_llama_factory_jsonl.py — 半自动数据蒸馏：组装与对齐
============================================================

读取人工（或大模型辅助）完成的 answer JSON 文件，与原始数据对齐，
生成 LLaMA-Factory 可直接训练的 ShareGPT 格式 JSONL。

输入:
  dataset/answers/
  ├── summarizer_batch_01_answers.json   # 人工填充的摘要 JSON 数组
  ├── summarizer_batch_02_answers.json
  ├── ...
  └── vision_answers.json                # 人工填充的视觉分析 JSON 数组

  dataset/tasks/
  ├── summarizer_batch_*.txt             # 原始作业文件 (含 data ID)
  └── vision_metadata.json               # Vision 原始元数据

输出:
  dataset/
  ├── summarizer_data.jsonl   # LLaMA-Factory Summarizer LoRA 训练集
  └── vision_data.jsonl       # LLaMA-Factory Vision LoRA 训练集 (含绝对图片路径)

运行方式:
  conda run -n ai_agent python scripts/build_llama_factory_jsonl.py

LLaMA-Factory ShareGPT 格式说明:
  每条数据为一行 JSON:
  {
    "messages": [
      {"role": "system",  "content": "系统提示词"},
      {"role": "user",    "content": "用户输入 (多模态用 <image> 标签)"},
      {"role": "assistant", "content": "期望输出"}
    ]
  }

  对于多模态 (Vision)，user content 中需包含图片的绝对路径:
  {"role": "user", "content": "<image>/absolute/path/to/vis_01.jpg</image>\n请分析..."}
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from typing import Optional

from config import Config

# =========================================================
# 路径与常量
# =========================================================
DATASET_DIR = PROJECT_ROOT / "dataset"
TASKS_DIR = DATASET_DIR / "tasks"
ANSWERS_DIR = DATASET_DIR / "answers"
VISION_IMG_DIR = TASKS_DIR / "vision_images"

SYSTEM_SUMMARIZER = (
    "You are a research assistant specialized in academic paper summarization. "
    "Given noisy text extracted from PDF pages, produce a concise, accurate "
    "Chinese summary (≤200 words). Preserve ALL model names, numerical metrics "
    "(mAP, AP50, F1 scores, etc.), and key experimental conclusions. Use Markdown."
)

SYSTEM_VISION = (
    "You are a vision-language academic analyst. Given a paper page image showing "
    "charts, tables, or figures, provide a detailed academic analysis in Chinese "
    "(≤300 words). Identify chart types, extract key numerical results, highlight "
    "trends, and explain the significance of findings in the paper context."
)


# =========================================================
# 1. Summarizer JSONL 构建
# =========================================================

def build_summarizer_jsonl():
    """
    读取所有 summarizer_batch_*_answers.json，与对应 batch .txt 中的 data ID
    对齐，生成 summarizer_data.jsonl。
    """
    print("\n" + "=" * 60)
    print("📝 [Summarizer] 组装 LLaMA-Factory JSONL")
    print("=" * 60)

    # 扫描所有 answer JSON 文件
    answer_files = sorted(ANSWERS_DIR.glob("summarizer_batch_*_answers.json"))
    if not answer_files:
        print("   ⚠️ 未找到 summarizer_batch_*_answers.json，请先填写答案")
        print(f"   预期位置: {ANSWERS_DIR}/")
        return

    all_samples = []
    errors = []

    for ans_file in answer_files:
        # 解析 batch 编号
        match = re.search(r"summarizer_batch_(\d+)_answers\.json", ans_file.name)
        batch_num = match.group(1) if match else "??"

        try:
            with open(ans_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{ans_file.name}: JSON 解析失败 — {e}")
            continue

        if not isinstance(answers, list):
            errors.append(f"{ans_file.name}: 格式错误，期望 JSON 数组")
            continue

        print(f"   处理 {ans_file.name}: {len(answers)} 条答案")

        for item in answers:
            if not isinstance(item, dict):
                errors.append(f"{ans_file.name}: 数组元素不是 dict")
                continue

            data_id = item.get("id", "")
            summary = item.get("summary", "")

            if not data_id or not summary:
                errors.append(f"{ans_file.name}: 缺少 'id' 或 'summary' 字段")
                continue

            # 从原始 batch 文件中读取对应的 page_text
            batch_txt = TASKS_DIR / f"summarizer_batch_{batch_num}.txt"
            page_text = _extract_page_text_from_batch(batch_txt, data_id)

            if page_text is None:
                errors.append(f"{ans_file.name}: 无法在 batch_{batch_num}.txt 中找到 ID={data_id}")
                continue

            all_samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_SUMMARIZER},
                    {"role": "user", "content": page_text},
                    {"role": "assistant", "content": summary},
                ]
            })

    if errors:
        print(f"   ⚠️ {len(errors)} 个错误:")
        for e in errors[:10]:
            print(f"      - {e}")
        if len(errors) > 10:
            print(f"      ... 还有 {len(errors) - 10} 个")

    if not all_samples:
        print("   ⚠️ 无有效样本，跳过")
        return

    # 去重 (按 page_text hash)
    seen = set()
    unique = []
    for s in all_samples:
        h = hash(s["messages"][1]["content"][:200])
        if h not in seen:
            seen.add(h)
            unique.append(s)

    # 写入 JSONL
    output_file = DATASET_DIR / "summarizer_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in unique:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   ✅ 生成 {len(unique)} 条 (去重后) → {output_file.name}")


def _extract_page_text_from_batch(batch_txt_path: Path, data_id: str) -> Optional[str]:
    """
    从 batch .txt 文件中提取指定 data_id 对应的原始 page_text。
    Batch 文件格式:
        [ID: 2403.02148_page5]
        noisy text content...
        ---

    如果 batch 文件不存在，尝试从其他 batch 文件查找。
    """
    if not batch_txt_path.exists():
        # 尝试搜索所有 batch 文件
        for alt_file in sorted(TASKS_DIR.glob("summarizer_batch_*.txt")):
            result = _parse_batch_file(alt_file, data_id)
            if result:
                return result
        return None

    return _parse_batch_file(batch_txt_path, data_id)


def _parse_batch_file(filepath: Path, target_id: str) -> Optional[str]:
    """解析单个 batch 文件，查找指定 ID 的文本"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    # 按 [ID: ...] 标记分割
    pattern = re.compile(r'\[ID:\s*([^\]]+)\]')
    matches = list(pattern.finditer(content))

    for i, match in enumerate(matches):
        current_id = re.sub(r'\s+', ' ', match.group(1).strip())
        if current_id != re.sub(r'\s+', ' ', target_id):
            continue

        # 提取这个 ID 到下一个 ID (或文件结束) 之间的文本
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        text = content[start:end]
        # 清理：去掉尾部的 "---" 分隔符和多余空白
        text = re.sub(r'\n---\s*$', '', text)
        text = text.strip()

        if len(text) > 50:  # 确保不是空段
            return text

    return None


# =========================================================
# 2. Vision JSONL 构建
# =========================================================

def build_vision_jsonl():
    """
    读取 vision_answers.json，与 vision_metadata.json 对齐，
    生成 vision_data.jsonl (含图片绝对路径)。
    """
    print("\n" + "=" * 60)
    print("👁️  [Vision] 组装 LLaMA-Factory JSONL")
    print("=" * 60)

    # 加载元数据
    metadata_file = TASKS_DIR / "vision_metadata.json"
    if not metadata_file.exists():
        print(f"   ⚠️ 未找到 {metadata_file.name}，请先运行 extract_to_worksheet.py")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_records = json.load(f)

    # 构建 image_id → metadata 映射
    meta_by_id = {r["image_id"]: r for r in metadata_records}

    # 加载答案
    answers_file = ANSWERS_DIR / "vision_answers.json"
    if not answers_file.exists():
        print(f"   ⚠️ 未找到 vision_answers.json，请先填写答案")
        print(f"   预期位置: {answers_file}")
        return

    try:
        with open(answers_file, "r", encoding="utf-8") as f:
            answers = json.load(f)
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 解析失败: {e}")
        return

    if not isinstance(answers, list):
        print("   ❌ 格式错误，期望 JSON 数组")
        return

    all_samples = []
    errors = []

    for item in answers:
        if not isinstance(item, dict):
            errors.append("数组元素不是 dict")
            continue

        image_id = item.get("image_id", "")
        analysis = item.get("analysis", "")

        if not image_id or not analysis:
            errors.append(f"缺少 image_id 或 analysis: {item}")
            continue

        # 查找对应元数据
        meta = meta_by_id.get(image_id)
        if meta is None:
            errors.append(f"image_id={image_id} 在 vision_metadata.json 中找不到")
            continue

        image_path = meta.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            errors.append(f"image_id={image_id} 图片不存在: {image_path}")
            continue

        # 构建表格上下文文本
        tables_md = ""
        for t_id, md_content in meta.get("page_tables", {}).items():
            tables_md += f"\n{t_id}:\n{md_content}\n"

        # 构建多模态 user content (含绝对路径)
        user_content = (
            f"<image>{image_path}</image>\n\n"
            f"请详细解析图中的对比数据。以下是从该页面提取的表格数据作为参考：\n\n"
            f"{tables_md}\n\n"
            f"Paper ID: {meta.get('paper_id', 'unknown')} | Page: {meta.get('page_number', '?')}"
        )

        all_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_VISION},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": analysis},
            ]
        })

    if errors:
        print(f"   ⚠️ {len(errors)} 个错误:")
        for e in errors[:10]:
            print(f"      - {e}")

    if not all_samples:
        print("   ⚠️ 无有效样本，跳过")
        return

    # 写入 JSONL
    output_file = DATASET_DIR / "vision_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   ✅ 生成 {len(all_samples)} 条 → {output_file.name}")


# =========================================================
# 主入口
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 VRAG 微调数据组装工具")
    print("=" * 60)

    # 检查必要目录
    if not ANSWERS_DIR.exists() or not list(ANSWERS_DIR.glob("*.json")):
        print("\n⚠️  dataset/answers/ 目录为空或不存在！")
        print("   请先将大模型生成的回答保存为 JSON 文件放入该目录。")
        print()
        print("   预期文件:")
        print("   - summarizer_batch_01_answers.json")
        print("   - summarizer_batch_02_answers.json")
        print("   - ...")
        print("   - vision_answers.json")
        print()
        print("   答案格式示例 (summarizer):")
        print('   [{"id": "2403.02148_page5", "summary": "该论文提出..."}]')
        print()
        print("   答案格式示例 (vision):")
        print('   [{"image_id": "vis_01", "analysis": "该表格展示了..."}]')

    try:
        build_summarizer_jsonl()
    except Exception as e:
        print(f"❌ Summarizer 组装失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        build_vision_jsonl()
    except Exception as e:
        print(f"❌ Vision 组装失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 数据组装完成！")
    print(f"   Summarizer: {DATASET_DIR / 'summarizer_data.jsonl'}")
    print(f"   Vision:     {DATASET_DIR / 'vision_data.jsonl'}")
    print(f"   Planner:    {DATASET_DIR / 'planner_data.jsonl'} (由 extract 脚本直接生成)")
    print()
    print("📋 LLaMA-Factory 使用方式:")
    print("   将 dataset/*.jsonl 注册到 LLaMA-Factory 的 dataset_info.json 中，")
    print("   指定对应的 LoRA 模块即可开始训练。")
    print("=" * 60)
