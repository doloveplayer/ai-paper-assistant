"""文本主路摄入管线 — 父子分块文本 + 按需视觉编码。

摄入策略:
  1. 文本始终以父子分块方式向量化 (BGE-M3)
  2. 视觉 (ColPali) 仅对含图表/表格的页面编码
  3. 写入新集合 vrag_text_primary，与旧管线 vrag_hybrid_collection 共存
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import uuid
import torch
import fitz
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter

import qdrant_client
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig,
    PointStruct, MultiVectorComparator,
    Filter, FieldCondition, MatchValue,
)

# Monkey Patch: 绕过 transformers PEFT _convert_peft_config_moe 对 PaliGemma 的 KeyError
import transformers
try:
    import transformers.integrations.peft
    transformers.integrations.peft._convert_peft_config_moe = (
        lambda peft_config, model_type: peft_config
    )
except Exception:
    pass

from colpali_engine.models import ColPali, ColPaliProcessor
from config import Config

# 复用 mix_ingest 的文本+表格提取函数 (不修改原文件)
from tools.mix_ingest import extract_text_and_tables_from_page

DATA_DIR = Config.DATA_DIR
CACHE_DIR = Config.CACHE_DIR
QDRANT_URL = Config.QDRANT_STORAGE_URL
COLLECTION_NAME = Config.TEXT_PRIMARY_COLLECTION

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("image_cache", exist_ok=True)


def _is_visual_page(page) -> bool:
    """规则判断: 页面包含表格 或 嵌入图片 → 视觉页"""
    tables = page.find_tables()
    if tables and len(tables.tables) >= Config.VISUAL_PAGE_MIN_TABLES:
        return True
    images = page.get_images()
    if len(images) >= Config.VISUAL_PAGE_MIN_IMAGES:
        return True
    return False


def _build_parent_text(chunks: list[str], child_idx: int) -> str:
    """拼接父块 = child_{i-1} + child_i + child_{i+1}"""
    w = Config.PARENT_CONTEXT_WINDOW
    start = max(0, child_idx - w)
    end = min(len(chunks), child_idx + w + 1)
    return "\n\n".join(chunks[start:end])


def _ensure_collection(client: qdrant_client.QdrantClient):
    """创建 vrag_text_primary 集合 (如不存在)"""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text_dense": VectorParams(
                    size=Config.BEG_EMBEDDING_SIZE,
                    distance=Distance.COSINE,
                ),
                "colpali_vision": VectorParams(
                    size=128,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )
        print(f"✅ 创建文本主路集合: {COLLECTION_NAME}")


def ingest_papers_text_primary(batch_size: int = 2):
    """批量摄入 DATA_DIR 中所有 PDF，支持断点续传。

    可独立运行: python tools/text_primary_ingest.py
    """
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ 未在 {DATA_DIR} 发现 PDF 文件。")
        return

    print("\n" + "=" * 60)
    print(f"📋 文本主路摄入 | {len(pdf_files)} 个 PDF | 集合: {COLLECTION_NAME}")
    print("=" * 60)
    for i, f in enumerate(pdf_files):
        mb = os.path.getsize(os.path.join(DATA_DIR, f)) / (1024 * 1024)
        print(f"  [{i+1}] {f} ({mb:.1f} MB)")

    confirm = input("\n⚠️ 即将加载 ColPali (GPU) + BGE-M3 (CPU)，请确保 vLLM 已关闭。\n确认 (y/n): ")
    if confirm.lower() != "y":
        print("已取消。")
        return

    # ---- 模型加载 ----
    print("\n⏳ 加载 ColPali 视觉模型 (GPU)...")
    model = ColPali.from_pretrained(
        Config.VISION_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    processor = ColPaliProcessor.from_pretrained(
        Config.VISION_MODEL_PATH, local_files_only=True
    )
    model.eval()

    print("⏳ 加载 BGE-M3 文本模型 (CPU)...")
    text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")

    print("⏳ 初始化 SentenceSplitter 分块器...")
    splitter = SentenceSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
    )

    print("🔌 连接 Qdrant...")
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    _ensure_collection(client)

    total_chunks = 0
    total_visual = 0

    for file_name in pdf_files:
        pdf_path = os.path.join(DATA_DIR, file_name)
        print(f"\n📄 {file_name}")

        # ---- 断点续传 ----
        ingested = 0
        try:
            f = Filter(must=[
                FieldCondition(key="file_name", match=MatchValue(value=file_name))
            ])
            ingested = client.count(
                collection_name=COLLECTION_NAME, count_filter=f
            ).count
        except Exception:
            pass

        page_images = convert_from_path(pdf_path)
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(page_images)

        if ingested > 0:
            print(f"   ⏭️ 已有 {ingested} 个 chunk，续传中...")

        points_batch = []

        for page_num in range(1, total_pages + 1):
            page_idx = page_num - 1
            page = pdf_doc[page_idx]

            # 1. 文本提取
            page_text, page_tables = extract_text_and_tables_from_page(page)

            # 2. 图表页判断
            is_visual = _is_visual_page(page)

            # 3. 视觉编码 (仅图表页)
            colpali_vector = None
            image_path = None
            if is_visual:
                total_visual += 1
                img = page_images[page_idx]
                image_path = os.path.abspath(
                    f"image_cache/{file_name}_page_{page_num}.jpg"
                )
                img.save(image_path, "JPEG")

                inputs = processor.process_images([img]).to("cuda")
                with torch.no_grad():
                    emb = model(**inputs)
                colpali_vector = emb[0].cpu().float().numpy().tolist()
                del inputs, emb
                torch.cuda.empty_cache()

            # 4. 文本分块
            if page_text.strip():
                child_chunks = splitter.split_text(page_text)
            else:
                child_chunks = ["[当前页面以图表为主，未提取到有效文本]"]
                page_text = child_chunks[0]

            # 5. 逐子块写入 Qdrant
            for ci, child_text in enumerate(child_chunks):
                parent_id = f"{file_name}_p{page_num}_c{ci}"
                parent_text = _build_parent_text(child_chunks, ci)

                child_embedding = text_model.encode(
                    child_text, normalize_embeddings=True
                ).tolist()

                vectors = {"text_dense": child_embedding}
                # 非视觉页用零向量占位 (Qdrant 所有点必须有同名向量)
                vectors["colpali_vision"] = (
                    colpali_vector if colpali_vector is not None
                    else [[0.0] * 128]
                )

                point = PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vectors,
                    payload={
                        "file_name": file_name,
                        "paper_id": file_name,
                        "page_number": page_num,
                        "chunk_index": ci,
                        "parent_id": parent_id,
                        "child_text": child_text,
                        "parent_text": parent_text,
                        "is_visual_page": is_visual,
                        "image_path": image_path,
                        "page_tables": page_tables if page_tables else {},
                    },
                )
                points_batch.append(point)
                total_chunks += 1

                if len(points_batch) >= batch_size * 4:
                    client.upsert(
                        collection_name=COLLECTION_NAME, points=points_batch
                    )
                    print(f"   📝 {total_chunks} chunks (视觉页: {total_visual})")
                    points_batch = []

        if points_batch:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)

        pdf_doc.close()
        print(f"   ✅ {file_name} 完成 ({total_pages} 页)")

    print(f"\n🏁 摄入完成 | 总子块: {total_chunks} | 其中视觉页: {total_visual}")


def ingest_paper_text_primary(paper_id: str, pdf_url: str) -> str:
    """Agent tool: 下载单篇论文 → 父子分块摄入 → vrag_text_primary。

    供 agent_core.py 工具注册使用。
    """
    from tools.vision_rag import robust_download_pdf
    import pikepdf

    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    _ensure_collection(client)

    # 查重
    try:
        f = Filter(should=[
            FieldCondition(key="paper_id", match=MatchValue(value=paper_id)),
            FieldCondition(
                key="file_name", match=MatchValue(value=f"{paper_id}.pdf")
            ),
        ])
        if client.count(collection_name=COLLECTION_NAME, count_filter=f).count > 0:
            return f"⏭️ 论文 {paper_id} 已存在于 {COLLECTION_NAME}，无需重复摄入。"
    except Exception:
        pass

    # 下载
    local_path = os.path.join(Config.DOWNLOAD_DIR, f"{paper_id}.pdf")
    if not robust_download_pdf(pdf_url, local_path):
        return f"❌ 下载失败: {pdf_url}"

    # 安全扫描
    try:
        with pikepdf.open(local_path):
            pass
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        return f"❌ PDF 安全扫描未通过: {e}"

    # 摄入
    try:
        model = ColPali.from_pretrained(
            Config.VISION_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        processor = ColPaliProcessor.from_pretrained(
            Config.VISION_MODEL_PATH, local_files_only=True
        )
        model.eval()
        text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")
        splitter = SentenceSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE,
            chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        )

        page_images = convert_from_path(local_path)
        pdf_doc = fitz.open(local_path)
        points_batch = []
        total = 0

        for page_num in range(1, len(page_images) + 1):
            page = pdf_doc[page_num - 1]
            page_text, page_tables = extract_text_and_tables_from_page(page)
            is_visual = _is_visual_page(page)

            colpali_vector = None
            image_path = None
            if is_visual:
                img = page_images[page_num - 1]
                image_path = os.path.join(
                    CACHE_DIR, f"{paper_id}_page_{page_num}.jpg"
                )
                img.save(image_path, "JPEG", quality=95)
                inputs = processor.process_images([img]).to("cuda")
                with torch.no_grad():
                    emb = model(**inputs)
                colpali_vector = emb[0].cpu().float().numpy().tolist()
                del inputs, emb
                torch.cuda.empty_cache()

            child_chunks = (
                splitter.split_text(page_text) if page_text.strip()
                else ["[图表页，无文本]"]
            )
            for ci, child_text in enumerate(child_chunks):
                parent_text = _build_parent_text(child_chunks, ci)
                child_embedding = text_model.encode(
                    child_text, normalize_embeddings=True
                ).tolist()

                vectors = {"text_dense": child_embedding}
                vectors["colpali_vision"] = (
                    colpali_vector if colpali_vector else [[0.0] * 128]
                )

                points_batch.append(PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vectors,
                    payload={
                        "file_name": f"{paper_id}.pdf",
                        "paper_id": paper_id,
                        "page_number": page_num,
                        "chunk_index": ci,
                        "parent_id": f"{paper_id}_p{page_num}_c{ci}",
                        "child_text": child_text,
                        "parent_text": parent_text,
                        "is_visual_page": is_visual,
                        "image_path": image_path,
                        "page_tables": page_tables or {},
                    },
                ))
                total += 1

            if len(points_batch) >= 8:
                client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
                points_batch = []
            torch.cuda.empty_cache()

        if points_batch:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)

        pdf_doc.close()
        return (
            f"✅ 论文 {paper_id} 摄入完成 ({len(page_images)} 页 → {total} 个子块)"
            f" → {COLLECTION_NAME}"
        )
    except Exception as e:
        return f"❌ 摄入异常: {e}"


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    ingest_papers_text_primary()
