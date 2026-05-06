# 微调数据蒸馏工作流

两阶段半自动数据蒸馏：挖矿 → 人工标注 → 组装。

## 工作流概览

```
[Qdrant + SQLite]
      │
      ▼  extract_to_worksheet.py (自动)
dataset/tasks/  ← 原始数据打包为作业文件
      │
      ▼  你 + 大模型 (GPT-4o / DeepSeek) 完成回答
dataset/answers/  ← JSON 格式答案
      │
      ▼  build_llama_factory_jsonl.py (自动)
dataset/*.jsonl  ← LLaMA-Factory 训练数据
```

---

## 第一步：挖矿 (extract_to_worksheet.py)

```bash
conda run -n ai_agent python scripts/extract_to_worksheet.py
```

**产出文件：**

| 文件 | 用途 |
|------|------|
| `dataset/tasks/summarizer_batch_01.txt` | 第 1 批 10 条带噪文本 + System Prompt |
| `dataset/tasks/summarizer_batch_02.txt` | 第 2 批 |
| `...` | 共 5 批 (50 条) |
| `dataset/tasks/vision_worksheet.md` | 视觉分析作业表 (含表格数据) |
| `dataset/tasks/vision_metadata.json` | 视觉元数据 (供组装脚本使用) |
| `dataset/tasks/vision_images/vis_*.jpg` | 复制的论文页面图片 |
| `dataset/planner_data.jsonl` | Planner 训练数据 (全自动，已可直接用) |

**配置：** `SUMMARIZER_COUNT=50`, `VISION_COUNT=30`, `BATCH_SIZE=10` (可在脚本开头修改)

---

## 第二步：标注 (人工 + 大模型辅助)

### Summarizer 标注

1. 将 `dataset/tasks/summarizer_batch_01.txt` 的内容粘贴给 GPT-4o / DeepSeek
2. 模型会返回一个 JSON 数组
3. 保存为 `dataset/answers/summarizer_batch_01_answers.json`
4. 重复以上步骤完成全部 5 个 batch

**答案 JSON 格式：**
```json
[
  {"id": "2403.02148_page5", "summary": "该论文提出了一种基于..."},
  {"id": "2310.12345_page3", "summary": "实验结果表明..."}
]
```

**重要：** `"id"` 字段必须与 .txt 文件中的 `[ID: ...]` 完全一致。

### Vision 标注

1. 打开 `dataset/tasks/vision_worksheet.md`
2. 对照每张图片 (`dataset/tasks/vision_images/vis_*.jpg`) 和表格数据，撰写分析
3. 将全部 30 条分析保存为 `dataset/answers/vision_answers.json`

**答案 JSON 格式：**
```json
[
  {"image_id": "vis_01", "analysis": "该表格展示了 YOLOv8 在雾天..."},
  {"image_id": "vis_02", "analysis": "..."}
]
```

---

## 第三步：组装 (build_llama_factory_jsonl.py)

```bash
conda run -n ai_agent python scripts/build_llama_factory_jsonl.py
```

**产出文件：**

| 文件 | 内容 | 样本数 |
|------|------|--------|
| `dataset/summarizer_data.jsonl` | Summarizer LoRA 训练集 | ≤50 |
| `dataset/vision_data.jsonl` | Vision LoRA 训练集 | ≤30 |
| `dataset/planner_data.jsonl` | Planner LoRA 训练集 | 自动提取 |

---

## LLaMA-Factory 配置

在 `dataset_info.json` 中注册：

```json
{
  "vrag_summarizer": {
    "file_name": "summarizer_data.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  },
  "vrag_vision": {
    "file_name": "vision_data.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  },
  "vrag_planner": {
    "file_name": "planner_data.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  }
}
```

---

## JSONL 输出格式

### Summarizer
```json
{"messages": [
  {"role": "system", "content": "You are a research assistant..."},
  {"role": "user", "content": "原始带噪 PDF 文本"},
  {"role": "assistant", "content": "精炼后的学术摘要 (Markdown)"}
]}
```

### Vision (多模态)
```json
{"messages": [
  {"role": "system", "content": "You are a vision-language..."},
  {"role": "user", "content": "<image>/absolute/path/vis_01.jpg</image>\n请分析图表..."},
  {"role": "assistant", "content": "详细的图表分析 (Markdown)"}
]}
```

### Planner
```json
{"messages": [
  {"role": "system", "content": "You are an academic research agent..."},
  {"role": "user", "content": "用户原始问题"},
  {"role": "assistant", "content": "[{\"name\": \"search_vision_knowledge\", \"args\": {...}}]"}
]}
```

---

## 目录结构总览

```
dataset/
├── tasks/
│   ├── summarizer_batch_01.txt      # 作业文件 (给大模型)
│   ├── summarizer_batch_02.txt
│   ├── ...
│   ├── vision_worksheet.md          # 视觉作业表
│   ├── vision_metadata.json         # 元数据 (给组装脚本)
│   └── vision_images/
│       ├── vis_01.jpg
│       ├── vis_02.jpg
│       └── ...
├── answers/                         # 人工/大模型填写的答案
│   ├── summarizer_batch_01_answers.json
│   ├── summarizer_batch_02_answers.json
│   ├── ...
│   └── vision_answers.json
├── summarizer_data.jsonl            # 最终训练数据
├── vision_data.jsonl
└── planner_data.jsonl
```
