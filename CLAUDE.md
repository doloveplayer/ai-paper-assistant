# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI-powered academic research agent with hybrid vision+text RAG for analyzing computer vision papers (focus: adverse-weather object detection). The agent can search Arxiv/Semantic Scholar, download papers, ingest them into a local vector database, and answer detailed questions about paper contents (figures, tables, formulas, methodology) using a ColPali + VLM pipeline.

## Environment

- Conda env: `ai_agent` (Python 3.10)
- Working directory: `/home/c2216-3090/disB/hyh/AI`

## Service startup order (all four required for full functionality)

```bash
# 1. Qdrant vector database (Docker)
sudo docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage:z qdrant/qdrant

# 2. vLLM text LLM on port 8000 (Qwen2.5-7B-Instruct-AWQ)
python -m vllm.entrypoints.openai.api_server \
  --model ./models/llm/qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b-instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.35 \
  --quantization awq --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 3. vLLM vision model on port 8001 (Qwen2-VL-2B-Instruct-AWQ)
python -m vllm.entrypoints.openai.api_server \
  --model ./models/llm/qwen/Qwen2-VL-2B-Instruct-AWQ \
  --served-model-name qwen2-vl-7b-instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.25 \
  --limit-mm-per-prompt '{"image": 3}' --enforce-eager --port 8001

# 4. Chainlit web UI on port 8053
python -m chainlit run app_ui.py --port 8053
```

CLI mode (no web UI): `python agent_core.py`

## Architecture

```
User (Chainlit UI / CLI)
  → agent_core.py: LangGraph state machine
    nodes: call_model → tools → call_model → ...
           summarize_conversation (triggered when >5000 tokens)
    edges: START → check_memory → agent ⇄ tools → END
    checkpoint: SQLite at chat_history/global_agent_memory.sqlite
```

The LLM (Qwen2.5-7B via vLLM on port 8000) is given three tools and decides which to call:

| Tool | File | Purpose |
|------|------|---------|
| `search_academic_papers` | `tools/vision_rag.py` | Web search via Arxiv + Semantic Scholar |
| `download_and_ingest_vision_paper` | `tools/vision_rag.py` | Download PDF → ColPali encode → Qdrant upsert |
| `search_vision_knowledge` | `tools/vision_rag.py` | Hybrid retrieval (ColPali + BGE text) → RRF fusion → LLM summarization → VLM analysis |

## Key files

- **`config.py`**: All paths, URLs, model names, and RAG hyperparameters. Single source of truth.
- **`agent_core.py`**: LangGraph graph definition, system prompt (SOP for the agent), memory compression, telemetry.
- **`app_ui.py`**: Chainlit web UI with background-thread agent execution to avoid WebSocket timeouts. Also auto-appends citation sources if the LLM omits them.
- **`tools/vision_rag.py`**: All three tools plus lazy-loaded models (ColPali 8-bit on GPU, BGE-M3 on CPU), Qdrant client, RRF fusion, text summarization, VLM image analysis.
- **`tools/mix_ingest.py`**: Standalone batch ingestion of local PDFs into the hybrid Qdrant collection. Also defines `extract_text_and_tables_from_page()` imported by `vision_rag.py`.

## Qdrant schema (hybrid collection: `vrag_hybrid_collection`)

Named vectors per point:
- `colpali_vision`: 128-dim multi-vector (ColPali late-interaction, `MAX_SIM` comparator)
- `text_dense`: 1024-dim dense vector (BGE-M3)

Payload fields: `paper_id`, `file_name`, `page_number`, `image_path`, `page_text`, `page_tables`

## Important constraints and non-obvious details

- **GPU memory is very tight.** ColPali (8-bit, ~3-4GB) + vLLM text (0.35 utilization) + vLLM vision (0.30) share one GPU. The lazy-loading pattern in `vision_rag.py` (`_get_vision_model()`, `_get_text_model()`, `_get_qdrant_client()`) delays GPU allocation until a tool is actually called. Never import models at module level.
- **Monkey patch required.** `transformers.integrations.peft._convert_peft_config_moe` fails with `KeyError: 'llava'` when ColPali (PaliGemma architecture) loads. The lambda patch in both `vision_rag.py` and `mix_ingest.py` skips this broken conversion. Do not remove without testing ColPali loading first.
- **vLLM is stateless per request.** KV cache is freed after each `/chat/completions` call. The `--max-model-len` flag is the per-request token limit, not cumulative. 3 images × ~2000 vision tokens ≈ 6000+ tokens, so `--max-model-len` must be ≥ 8192 for the VLM.
- **PDF pages are rendered one at a time** (`convert_from_path` with `first_page`/`last_page`) to prevent OOM on long PDFs.
- **RRF (Reciprocal Rank Fusion)** with `k=60` merges ColPali vision results with BGE text results. Candidate pool is `top_k * 6` before RRF truncation.
- **Text summarization before VLM**: `_summarize_payloads()` calls the local LLM to compress multi-page text into a ~500-character Chinese summary, saving ~90% of text tokens sent to the VLM.
- **Citation sources**: All tools append `📚 引用来源` blocks to their output. `app_ui.py` has a fallback that auto-appends them if the LLM's final response lacks them. The system prompt (rule 4) requires the LLM to include sources.

## Running ingestion

```bash
# Hybrid ingest (ColPali vision + BGE text) — recommended
python tools/mix_ingest.py

# Vision-only ingest (ColPali only)
python tools/vision_ingest.py

# Text-only ingest (legacy LlamaIndex pipeline)
python tools/text_ingest.py
```

Place PDFs in `data/` before running. `mix_ingest.py` supports resumable ingestion (checks existing page count per file before starting).

## Model download

```bash
python download_models.py   # BGE-M3 from ModelScope
python download_llm.py      # Qwen2-VL-2B-Instruct-AWQ from ModelScope
```
