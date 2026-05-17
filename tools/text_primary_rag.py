"""文本主路检索管线 — 父子分块 BGE-M3 主查 + ColPali 视觉辅助 + VLM 分析。

检索流程:
  1. BGE-M3 子块检索 (pool_size=15) → 按 parent_id 去重
  2. ColPali 视觉页检索 (pool_size=6, filter is_visual_page==True)
  3. 加权 RRF 融合 (text=0.7, vision=0.3, k=60)
  4. 父块文本 → LLM 摘要
  5. 视觉命中 → VLM 图文分析 | 纯文本 → 摘要即最终回答
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import base64
import torch
import requests
import numpy as np
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from PIL import Image

import qdrant_client
from qdrant_client.models import (
    Filter, FieldCondition, MatchAny, MatchText, MatchValue,
)

# Monkey Patch
import transformers
try:
    import transformers.integrations.peft
    transformers.integrations.peft._convert_peft_config_moe = (
        lambda peft_config, model_type: peft_config
    )
except Exception:
    pass

from colpali_engine.models import ColPali, ColPaliProcessor

# 复用视觉分析函数 (不修改 vision_rag.py)
from tools.vision_rag import (
    _compute_similarity_heatmap,
    _heatmap_to_roi,
    _stitch_rois_to_composite,
)
from config import Config

# ---- 懒加载 (按需初始化，避免 import 即占 GPU) ----
_model = None
_processor = None
_text_model = None
_client = None

COLLECTION_NAME = Config.TEXT_PRIMARY_COLLECTION


def _get_vision_model():
    global _model, _processor
    if _model is None:
        print("⏳ [TextPrimary] 加载 ColPali (8-bit)...")
        from transformers import BitsAndBytesConfig
        q_config = BitsAndBytesConfig(load_in_8bit=True)
        _model = ColPali.from_pretrained(
            Config.VISION_MODEL_PATH,
            quantization_config=q_config,
            device_map="cuda",
            local_files_only=True,
        )
        _processor = ColPaliProcessor.from_pretrained(
            Config.VISION_MODEL_PATH, local_files_only=True
        )
        _model.eval()
    return _model, _processor


def _get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")
    return _text_model


def _get_client():
    global _client
    if _client is None:
        _client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)
    return _client


# ---- 加权 RRF 融合 ----
def _weighted_rrf_fusion(
    text_hits: list,
    vision_hits: list,
    k: int = 60,
    w_text: float = 0.7,
    w_vision: float = 0.3,
):
    """加权倒数排名融合。text_hits 和 vision_hits 是 ScoredPoint 列表。"""
    scores = {}
    hit_dict = {}

    for rank, hit in enumerate(text_hits):
        scores[hit.id] = scores.get(hit.id, 0) + w_text / (k + rank + 1)
        hit_dict[hit.id] = hit

    for rank, hit in enumerate(vision_hits):
        scores[hit.id] = scores.get(hit.id, 0) + w_vision / (k + rank + 1)
        hit_dict[hit.id] = hit

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [hit_dict[doc_id] for doc_id in sorted_ids]


# ---- 父块文本摘要 ----
def _summarize_parents(query: str, parents: list[dict]) -> str:
    """用 LLM 将检索到的父块压缩为聚焦查询的摘要 (中文 ≤500 字)。"""
    if not parents:
        return ""

    texts = []
    for p in parents:
        header = (
            f"--- 文献: {p['doc_id']} | 页码: {p['page_num']}"
            f" | 块: {p['chunk_idx']} ---"
        )
        texts.append(f"{header}\n{p['parent_text'][:3000]}")

    prompt = (
        f"你是一个学术论文分析助手。用户的问题是：'{query}'\n\n"
        f"以下是检索到的论文上下文片段（每个片段已包含前后文）。"
        f"请提取与用户问题直接相关的信息，生成一份简洁的摘要"
        f"（中文，总计不超过500字）：\n\n"
        + "\n\n".join(texts)
    )

    try:
        resp = requests.post(
            f"{Config.LLM_API_BASE}/chat/completions",
            json={
                "model": Config.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"   ⚠️ 摘要失败: {e}")

    # Fallback: raw text
    return "\n\n".join(
        f"--- {p['doc_id']} p{p['page_num']} ---\n{p['parent_text'][:500]}"
        for p in parents
    )


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---- Tool Input Schema ----
class SearchTextPrimaryInput(BaseModel):
    query: str = Field(description="检索查询，建议使用英文关键词")
    paper_id: Optional[str] = Field(
        default=None,
        description="论文 ID (如 2403.02148)，指定则仅在该论文内检索",
    )


@tool("search_text_primary_knowledge", args_schema=SearchTextPrimaryInput)
def search_text_primary_knowledge(query: str, paper_id: Optional[str] = None) -> str:
    """
    【文本主路深度研读工具】
    优先使用此工具检索论文的文本内容（方法论、实验参数、公式、数据集描述等）。
    采用父子分块检索 —— 用小粒度子块精准匹配，用大粒度父块提供完整上下文。
    仅在命中图表页时启用视觉分析作为辅助。
    """
    top_k = 3
    print(
        f"\n🔍 [TextPrimary] 检索: '{query}' "
        + (f"| 文献: {paper_id}" if paper_id else "| 全局")
    )

    text_model = _get_text_model()
    client = _get_client()

    # ---- 1. 文本查询向量 ----
    text_query = text_model.encode(query, normalize_embeddings=True).tolist()

    # ---- 2. Paper ID 过滤器 (复用 vision_rag.py 的鲁棒匹配模式) ----
    query_filter = None
    if paper_id:
        pid_variants = [paper_id]
        if paper_id.endswith(".pdf"):
            pid_variants.append(paper_id[:-4])
        else:
            pid_variants.append(paper_id + ".pdf")
        clean = paper_id.strip().removesuffix(".pdf").replace("\n", " ")
        prefix = clean[:40] if len(clean) > 40 else clean
        query_filter = Filter(should=[
            FieldCondition(key="paper_id", match=MatchAny(any=pid_variants)),
            FieldCondition(key="file_name", match=MatchAny(any=pid_variants)),
            FieldCondition(key="file_name", match=MatchText(text=prefix)),
        ])

    # ---- 3. 文本路检索 (主路) ----
    text_res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=text_query,
        using="text_dense",
        query_filter=query_filter,
        limit=Config.TEXT_SEARCH_POOL_SIZE,
    ).points

    # 按 parent_id 去重，保留每个 parent 最高分的子块
    parent_best: dict = {}
    for hit in text_res:
        pid = hit.payload.get("parent_id", hit.id)
        if pid not in parent_best or hit.score > parent_best[pid].score:
            parent_best[pid] = hit

    deduped_text_hits = sorted(
        parent_best.values(), key=lambda h: h.score, reverse=True
    )

    # ---- 4. 视觉路检索 (辅路，仅 is_visual_page==True) ----
    model, processor = _get_vision_model()
    vision_inputs = processor.process_queries([query]).to("cuda")
    if "token_type_ids" not in vision_inputs:
        vision_inputs["token_type_ids"] = torch.zeros_like(
            vision_inputs["input_ids"]
        )
    with torch.no_grad():
        query_emb = model(**vision_inputs)
    vision_query = query_emb[0].cpu().float().numpy().tolist()

    # 构建视觉路过滤: is_visual_page==True + paper_id filter
    if paper_id:
        vision_filter = Filter(must=[
            FieldCondition(key="is_visual_page", match=MatchValue(value=True)),
            FieldCondition(key="paper_id", match=MatchAny(any=pid_variants)),
        ])
    else:
        vision_filter = Filter(must=[
            FieldCondition(key="is_visual_page", match=MatchValue(value=True)),
        ])

    vision_res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vision_query,
        using="colpali_vision",
        query_filter=vision_filter,
        limit=Config.VISION_SEARCH_POOL_SIZE,
    ).points

    # ---- 5. 加权 RRF 融合 ----
    fused = _weighted_rrf_fusion(
        text_hits=list(deduped_text_hits),
        vision_hits=vision_res,
        k=60,
        w_text=Config.RRF_TEXT_WEIGHT,
        w_vision=Config.RRF_VISION_WEIGHT,
    )
    fused = fused[:top_k]

    print(
        f"👁️ [TextPrimary] 文本命中 {len(deduped_text_hits)} 父块, "
        f"视觉命中 {len(vision_res)}, 融合后 top {len(fused)}"
    )

    if not fused:
        return "未在知识库中找到相关片段。"

    # ---- Eval 注入 (与 vision_rag.py 保持一致的 side-channel 模式) ----
    try:
        from eval.eval_tracer import EvalContext
        ctx = EvalContext.get()
        if ctx and ctx.enabled:
            ctx.capture_retrieval(
                query=query,
                paper_id=paper_id,
                vision_hits=[
                    {"id": h.id, "score": h.score, "payload": dict(h.payload)}
                    for h in vision_res
                ],
                text_hits=[
                    {"id": h.id, "score": h.score, "payload": dict(h.payload)}
                    for h in list(deduped_text_hits)
                ],
                fused_hits=[
                    {"id": h.id, "score": 0.0, "payload": dict(h.payload)}
                    for h in fused
                ],
            )
    except Exception:
        pass

    # ---- 6. 组装父块 + 视觉信息 ----
    parents = []
    visual_pages = []
    seen_parents = set()

    for hit in fused:
        payload = dict(hit.payload)
        parent_id = payload.get("parent_id", hit.id)
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        doc_id = payload.get("paper_id", payload.get("file_name", "未知"))
        page_num = payload.get("page_number", 0)
        chunk_idx = payload.get("chunk_index", 0)

        parents.append({
            "doc_id": doc_id,
            "page_num": page_num,
            "chunk_idx": chunk_idx,
            "parent_text": payload.get(
                "parent_text", payload.get("child_text", "")
            ),
        })

        if payload.get("is_visual_page") and payload.get("image_path"):
            visual_pages.append({
                "doc_id": doc_id,
                "page_num": page_num,
                "image_path": payload["image_path"],
                "page_tables": payload.get("page_tables", {}),
                "child_text": payload.get("child_text", ""),
            })

    # ---- 7. 文本摘要 ----
    summary = _summarize_parents(query, parents)

    # ---- 8. VLM 分析 (仅当有视觉命中时) ----
    has_visual = len(visual_pages) > 0

    if has_visual:
        vision_messages = [{
            "type": "text",
            "text": (
                f"你是一个严谨的学术分析员。用户的问题是：'{query}'。\n"
                f"请结合裁剪后的关键区域图片和文本摘要综合分析并回答。\n"
                f"【护栏规则】:\n"
                f"1. 文本摘要中的数值、表格数据可作为事实依据。\n"
                f"2. 严禁编造信息，必须说明引用出处。\n\n"
                f"【文本检索摘要】:\n{summary}"
            ),
        }]

        total_roi = 0
        total_fallback = 0

        for vp in visual_pages:
            img_path = vp["image_path"]
            if not img_path or not os.path.exists(img_path):
                continue

            try:
                heatmap, orig_w, orig_h = _compute_similarity_heatmap(
                    query_embeddings=query_emb,
                    image_path=img_path,
                    model=model,
                    processor=processor,
                    device="cuda",
                )
                rois = _heatmap_to_roi(
                    heatmap, orig_w, orig_h,
                    n_patches=(32, 32), top_k=5, context_padding=0.25,
                )
                del heatmap
                torch.cuda.empty_cache()

                if rois:
                    b64s = _stitch_rois_to_composite(img_path, rois)
                    for b64 in b64s:
                        vision_messages.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                    total_roi += len(rois)
                else:
                    b64 = _encode_image_to_base64(img_path)
                    vision_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                    total_fallback += 1
            except Exception as e:
                print(f"   ⚠️ 视觉提取异常: {e}")

        del query_emb
        torch.cuda.empty_cache()

        if total_roi > 0:
            print(
                f"📐 [TextPrimary] {total_roi} ROI + {total_fallback} 回退全图, "
                f"预计节省 ~{total_roi * 60}% VLM token"
            )

        try:
            resp = requests.post(
                f"{Config.VLLM_API_BASE}/chat/completions",
                json={
                    "model": Config.VLLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": vision_messages}],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                body = resp.json()["choices"][0]["message"]["content"]
            else:
                body = summary + f"\n\n⚠️ VLM 返回 {resp.status_code}"
        except Exception as e:
            body = summary + f"\n\n⚠️ VLM 调用失败: {e}"
    else:
        del query_emb
        torch.cuda.empty_cache()
        body = summary
        print("   📝 无视觉命中，纯文本回答模式")

    # ---- 9. 引用来源 ----
    seen_src = set()
    src_lines = []
    for p in parents:
        key = (p["doc_id"], p["page_num"])
        if key not in seen_src:
            seen_src.add(key)
            src_lines.append(
                f"- 文献: `{p['doc_id']}` | 页码: {p['page_num']}"
                f" (本地RAG — text_primary)"
            )

    return (
        body
        + "\n\n---\n📚 **引用来源 (本地RAG - 文本主路)**:\n"
        + "\n".join(src_lines)
    )
