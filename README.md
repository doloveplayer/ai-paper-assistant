# VRAG — Vision-Augmented Academic Research Agent

基于视觉+文本双路 RAG 的学术论文智能分析助手。支持全网论文检索、自动下载入库、以及结合图表/公式/表格的多模态深度研读。

## 功能概览

1. **全网检索** — 通过 Arxiv + Semantic Scholar 搜索最新论文，自动过滤近 3 年 CV 领域前沿工作
2. **自动入库** — 下载 PDF → 安全扫描 → ColPali 视觉编码 + BGE-M3 文本编码 → Qdrant 双路向量入库
3. **深度研读** — 混合检索（RRF 融合）→ 文本摘要压缩 → VLM 多模态图文分析，精准回答论文细节

## 项目结构

```
.
├── config.py              # 全局配置中心（路径、模型、RAG 参数）
├── agent_core.py          # LangGraph 状态机（Agent 核心逻辑 + 记忆压缩）
├── app_ui.py              # Chainlit Web 界面
├── download_models.py     # 下载 BGE-M3 嵌入模型
├── download_llm.py        # 下载 Qwen2-VL 视觉语言模型
├── eval_system.py         # RAG 评估框架
├── env.txt                # 环境安装命令记录
├── CLAUDE.md              # Claude Code 开发指南
└── tools/
    ├── vision_rag.py       # 三个 Agent 工具（搜索/入库/检索）+ 模型懒加载
    ├── mix_ingest.py       # 批量双路 PDF 入库脚本
    ├── vision_ingest.py    # 纯视觉入库脚本
    ├── text_ingest.py      # 纯文本入库脚本（LlamaIndex）
    └── delete_qdrant_data.py  # Qdrant 数据清理
```

## 技术架构

```
User (Chainlit / CLI)
  → agent_core.py: LangGraph 状态机
      → LLM (Qwen2.5-7B, vLLM :8000) 决定调用工具
          ├── search_academic_papers     → Arxiv + Semantic Scholar
          ├── download_and_ingest_paper  → 下载 + ColPali/BGE 双路编码 + Qdrant 入库
          └── search_vision_knowledge   → 双路检索 + RRF 融合 + LLM 摘要 + VLM 图文分析
      → summarize_conversation (Token > 5000 时自动压缩长期记忆)
      → SQLite 持久化对话历史
```

### 双路向量存储 (Qdrant `vrag_hybrid_collection`)

| 通道 | 模型 | 维度 | 类型 |
|------|------|------|------|
| `colpali_vision` | ColPali (8-bit) | 128 | Multi-vector (MAX_SIM) |
| `text_dense` | BGE-M3 | 1024 | Dense (COSINE) |

## 环境要求

- **Python**: 3.10
- **GPU**: NVIDIA GPU (建议 24GB+ 显存)
- **CUDA**: 12.4
- **系统**: Linux (Ubuntu 20.04+)

### 依赖安装

```bash
# 创建环境
conda create -n ai_agent python=3.10 -y
conda activate ai_agent

# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 核心库
pip install openai langchain langgraph chainlit vllm
pip install llama-index llama-index-embeddings-huggingface llama-index-vector-stores-qdrant
pip install qdrant-client sentence-transformers transformers
pip install PyMuPDF pikepdf pdf2image arxiv pydantic
pip install colpali-engine modelscope
```

### 模型下载

```bash
python download_models.py    # BGE-M3 嵌入模型 (~2GB)
python download_llm.py       # Qwen2-VL 视觉模型 (~15GB)
```

另外需自行下载：
- **Qwen2.5-7B-Instruct-AWQ**: 放置到 `models/llm/qwen/Qwen2.5-7B-Instruct-AWQ/`
- **ColPali v1.2**: 放置到 `models/colpali-v1.2/`

## 启动运行

需依次启动 4 个服务：

```bash
# 1. Qdrant 向量数据库
sudo docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage:z qdrant/qdrant

# 2. vLLM 文本 LLM (端口 8000)
python -m vllm.entrypoints.openai.api_server \
  --model ./models/llm/qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b-instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.35 \
  --quantization awq --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 3. vLLM 视觉模型 (端口 8001)
python -m vllm.entrypoints.openai.api_server \
  --model ./models/llm/qwen/Qwen2-VL-2B-Instruct-AWQ \
  --served-model-name qwen2-vl-7b-instruct \
  --max-model-len 12288 --gpu-memory-utilization 0.30 \
  --limit-mm-per-prompt '{"image": 3}' --enforce-eager --port 8001

# 4. Chainlit Web UI (端口 8053)
python -m chainlit run app_ui.py --port 8053
```

CLI 模式（无需启动 Chainlit）：
```bash
python agent_core.py
```

## 数据入库

将 PDF 论文放入 `data/` 目录后执行：

```bash
python tools/mix_ingest.py    # 视觉+文本双路入库（推荐）
python tools/vision_ingest.py # 纯视觉入库
python tools/text_ingest.py   # 纯文本入库（LlamaIndex）
```

## 重要说明

- **GPU 显存紧张**：ColPali (~4GB) + vLLM 文本 + vLLM 视觉共用一张 GPU，模型采用懒加载模式延迟分配
- **vLLM 无状态**：每次请求结束后 KV Cache 自动释放，`--max-model-len` 为单次请求上限
- **逐页渲染 PDF**：防止长 PDF 导致 OOM
- **Monkey Patch**：transformers PEFT 对 PaliGemma 架构存在兼容性问题，`vision_rag.py` 和 `mix_ingest.py` 中已内置热修复
- 首次使用前需在 `config.py` 中确认所有路径与实际环境一致

## License

MIT
