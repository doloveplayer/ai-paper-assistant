import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import time
import uuid
import arxiv
import requests
import fitz
from sentence_transformers import SentenceTransformer
import torch
import pikepdf
import base64
from typing import Optional
from langchain_core.tools import tool
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
import qdrant_client 
from qdrant_client.models import Distance, VectorParams, MultiVectorConfig, PointStruct, MultiVectorComparator, Filter, FieldCondition, MatchAny, MatchValue

from tools.mix_ingest import extract_text_and_tables_from_page 

# 引入全局配置
from config import Config

# =========================================================
# 🚨 终极热修复 (Monkey Patch)：直接拦截并废除官方的字符串转换 Bug
# TODO: transformers 某些版本中 _convert_peft_config_moe 对 MoE 配置做了错误的
# str() 转换导致加载失败。此 lambda 直接透传原始配置跳过转换。
# 触发条件: ColPali + BitsAndBytesConfig + PEFT 混用时。可在升级 transformers
# 后尝试移除此 patch 看是否修复。
# =========================================================
import transformers
try:
    import transformers.integrations.peft
    transformers.integrations.peft._convert_peft_config_moe = lambda peft_config, model_type: peft_config
except Exception:
    pass

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings
from transformers import BitsAndBytesConfig
from PIL import Image
import io
import numpy as np

# ==========================================
# 1. 模型与客户端 —— 懒加载 (按需初始化，避免 import 即占 GPU)
# ==========================================
_model = None
_processor = None
_text_model = None
_client = None

TEXT_DIM = Config.BEG_EMBEDDING_SIZE
collection_name = Config.COLLECTION_MIX_NAME

os.makedirs(Config.DOWNLOAD_DIR, exist_ok=True)
CACHE_DIR = Config.CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_vision_model():
    global _model, _processor
    if _model is None:
        print("⏳ [Tool] 正在挂载 ColPali 视觉向量引擎 (8-bit 量化常驻版)...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        _model = ColPali.from_pretrained(
            Config.VISION_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="cuda",
            local_files_only=True
        )
        _processor = ColPaliProcessor.from_pretrained(Config.VISION_MODEL_PATH, local_files_only=True)
        _model.eval()
    return _model, _processor


def _get_text_model():
    global _text_model
    if _text_model is None:
        print("⏳ [Tool] 正在挂载 BGE 文本特征提取器 (纯 CPU)...")
        _text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")
    return _text_model


def _get_qdrant_client():
    global _client
    if _client is None:
        _client = qdrant_client.QdrantClient(url=Config.QDRANT_STORAGE_URL)
        if not _client.collection_exists(collection_name):
            _client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "colpali_vision": VectorParams(
                        size=128,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        )
                    ),
                    "text_dense": VectorParams(
                        size=TEXT_DIM,
                        distance=Distance.COSINE
                    )
                }
            )
    return _client

# ==========================================
# 3. 核心基建：安全扫描与下载
# ==========================================
def robust_download_pdf(url, save_path, max_retries=3):
    """工业级流式下载函数，带断线重试机制"""
    for attempt in range(max_retries):
        try:
            print(f"   🌐 正在建立网络连接拉取 PDF... (尝试 {attempt + 1}/{max_retries})")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"   ⚠️ 下载中断: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return False

def is_suspicious_action(action_obj) -> bool:
    if not isinstance(action_obj, pikepdf.Dictionary): return False
    if '/S' in action_obj and str(action_obj['/S']) in ['/JavaScript', '/Launch', '/SubmitForm', '/ImportData']: return True
    if '/JS' in action_obj: return True
    return False

def scan_pdf_for_malware(file_path: str) -> tuple[bool, str]:
    """工业级 PDF 恶意代码物理拦截器"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:
            return False, f"文件过大 ({file_size / (1024*1024):.2f} MB)，已拒绝。"
        with pikepdf.Pdf.open(file_path) as pdf:
            root = pdf.Root
            if '/Names' in root and '/JavaScript' in root['/Names']: return False, "检测到全局 JS 注入。"
            if '/OpenAction' in root and is_suspicious_action(root['/OpenAction']): return False, "检测到恶意触发器。"
            for page in pdf.pages:
                if '/AA' in page:
                    for trigger, action in page['/AA'].items():
                        if is_suspicious_action(action): return False, f"页面藏有触发脚本。"
        return True, "安全扫描通过。"
    except Exception as e:
        return False, f"扫描引擎异常或文件损坏: {str(e)}"

# ==========================================
# 4. Agent 统一工具集
# ==========================================

class AcademicSearchInput(BaseModel):
    primary_concept: str = Field(description="核心研究任务/算法名。例如 'object detection' 或 'YOLO'。必须使用纯英文！")
    secondary_concept: str = Field(description="具体的限定场景或前置条件。例如 'adverse weather' 或 'domain adaptation'。必须使用纯英文！")
    max_results: int = Field(default=5, description="最大返回数量")
@tool("search_academic_papers", args_schema=AcademicSearchInput)
def search_academic_papers(primary_concept: str, secondary_concept: str, max_results: int = 5) -> str:
    """
    【学术检索引擎】当用户查找前沿论文时调用。
    """
    # 组合给 Semantic Scholar 用的宽泛词
    general_query = f"{primary_concept} {secondary_concept}"
    
    # ==========================================
    # ⚡ 核心黑魔法：构建 Arxiv 高级检索语法
    # 语法解析：
    # 1. abs:"xxx" 代表【严格匹配】摘要中的这个词组，少一个字母都不行。
    # 2. AND 必须全大写，代表必须同时满足。
    # 3. cat:cs.CV 代表【强行限定】在 Computer Science -> Computer Vision 领域内搜索！
    # ==========================================
    arxiv_advanced_query = f'abs:"{primary_concept}" AND abs:"{secondary_concept}" AND cat:cs.CV'
    
    print(f"\n🔍 [Agent 后台] 发起精准学术检索:")
    print(f"   - 语义意图: '{general_query}'")
    print(f"   - Arxiv 编译后底层指令: '{arxiv_advanced_query}'")
    
    results_pool = []
    
    # 策略 1: 优先 Arxiv 精准检索
    try:
        # 注意：这里传入的是编译后的高级语法 arxiv_advanced_query
        search = arxiv.Search(
            query=arxiv_advanced_query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.SubmittedDate # 既然关键词绝对精准了，我们优先看最新的论文！
        )
        for result in search.results():
            arxiv_id = result.get_short_id()
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            results_pool.append(
                f"Paper ID: {arxiv_id}\nTitle: {result.title}\n"
                f"📚 来源: Arxiv (网页检索)\n"
                f"🔗 PDF链接: {pdf_url}\nSummary: {result.summary[:200]}...\n---"
            )
    except Exception as e:
        print(f"   ⚠️ Arxiv 检索波动: {e}")

    # 策略 2: Semantic Scholar 兜底 (它自带基于 AI 的语义理解，直接传 general_query 即可)
    if len(results_pool) < max_results:
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            # 增加 year 限定，只找近 3 年的论文，保证前沿性
            params = {
                "query": general_query, 
                "limit": max_results, 
                "fields": "paperId,title,abstract,openAccessPdf",
                "year": "2023-" 
            }
            response = requests.get(url, params=params, timeout=10)
            # ... (这部分保留你之前的 Semantic Scholar 解析代码) ...
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    # 避免与 Arxiv 结果重复
                    if any(item.get("title", "")[:20].lower() in res.lower() for res in results_pool):
                        continue
                        
                    pdf_info = item.get("openAccessPdf")
                    if pdf_info and isinstance(pdf_info, dict) and "url" in pdf_info:
                        pdf_url = pdf_info['url']
                        results_pool.append(
                            f"Paper ID: {item.get('paperId')}\nTitle: {item.get('title')}\n"
                            f"📚 来源: Semantic Scholar (网页检索)\n"
                            f"🔗 PDF链接: {pdf_url}\nSummary: {str(item.get('abstract'))[:200]}...\n---"
                        )
                    if len(results_pool) >= max_results: break
        except Exception as e:
            print(f"   ⚠️ Semantic Scholar 补充检索波动: {e}")

    if not results_pool:
        return "未能找到严格匹配该领域的最新开源论文，请尝试使用更宽泛的学术缩写（如将 'adverse weather' 改为 'fog' 或 'rain'）。"

    results_text = "\n".join(results_pool)
    results_text += "\n\n---\n📚 **引用来源 (网页检索)**: 以上所有论文均来自 Arxiv 或 Semantic Scholar 的公开检索结果，PDF链接可直接访问。"
    return results_text

class IngestVisionPaperInput(BaseModel):
    paper_id: str = Field(description="论文的唯一标识符 (Paper ID)")
    pdf_url: str = Field(description="直接指向该论文 PDF 文件的公开链接 (PDF URL)")
@tool("download_and_ingest_vision_paper", args_schema=IngestVisionPaperInput)
def download_and_ingest_vision_paper(paper_id: str, pdf_url: str) -> str:
    """
    【统一视觉入库工具】当你通过搜索工具得到了一篇极其重要的论文的 Paper ID 和 PDF URL 后，调用此工具。
    它会自动执行：下载 -> 安全防爆扫描 -> 视觉多模态特征提取 -> Qdrant 永久入库。
    """
    print(f"\n📥 [Agent 后台] 启动视觉入库流 | ID: {paper_id}")

    # 1. 查重：是否已存在
    try:
        exact_match_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
        if _get_qdrant_client().count(collection_name=collection_name, count_filter=exact_match_filter).count > 0:
            return f"拦截：论文 {paper_id} 已存在于视觉知识库中，请直接调用检索工具，无需重复消耗算力。"
    except Exception as e:
        print(f"   ⚠️ 查重检查失败: {e}，继续入库流程。")

    # 2. 下载与安全扫描
    local_pdf_path = os.path.join(Config.DOWNLOAD_DIR, f"{paper_id}.pdf")
    if not robust_download_pdf(pdf_url, local_pdf_path):
        return f"失败：多次尝试从 {pdf_url} 拉取 PDF 失败，可能存在反爬虫限制。"

    safe, msg = scan_pdf_for_malware(local_pdf_path)
    if not safe:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        return f"严重拦截：该 PDF 未通过底层安全扫描 ({msg})，已强制销毁。"

    # 3. ColPali 视觉编码入库 (逐页渲染，防长 PDF OOM)
    try:
        print("   👁️  正在唤醒 双路处理管道 (ColPali + PyMuPDF)...")

        pdf_doc = fitz.open(local_pdf_path)
        total_pages = pdf_doc.page_count
        model, processor = _get_vision_model()
        text_model = _get_text_model()
        client = _get_qdrant_client()

        for page_num in range(1, total_pages + 1):
            images = convert_from_path(local_pdf_path, first_page=page_num, last_page=page_num)

            # [路 1：视觉特征]
            inputs = processor.process_images(images).to("cuda")
            with torch.no_grad():
                image_embeddings = model(**inputs)

            vision_embedding = image_embeddings[0]
            point_id = uuid.uuid4().hex

            # [路 2：文本清洗与特征提取]
            page_text, page_tables = extract_text_and_tables_from_page(pdf_doc[page_num - 1])
            if page_text.strip():
                text_embedding = text_model.encode(page_text, normalize_embeddings=True)
            else:
                text_embedding = [0.0] * TEXT_DIM
                page_text = "[本页文本信息提取为空，主要依靠视觉内容]"

            img_save_path = os.path.join(CACHE_DIR, f"{paper_id}_page_{page_num}.jpg")
            images[0].save(img_save_path, "JPEG", quality=95)

            point = PointStruct(
                id=point_id,
                vector={
                    "colpali_vision": vision_embedding.cpu().float().numpy().tolist(),
                    "text_dense": text_embedding.tolist()
                },
                payload={
                    "paper_id": paper_id,
                    "page_number": page_num,
                    "image_path": img_save_path,
                    "page_text": page_text,
                    "page_tables": page_tables
                }
            )
            client.upsert(collection_name=collection_name, points=[point])
            print(f"   ⚙️ 双模态映射进度: {page_num} / {total_pages} 页")

            del inputs
            del image_embeddings
            torch.cuda.empty_cache()

        pdf_doc.close()
        return f"✅ 完美融合！论文 {paper_id} (共 {total_pages} 页) 的【视觉+文本双维矩阵】已永久写入 Qdrant。"

    except Exception as e:
        return f"入库崩溃：多模态编码过程中发生异常 - {str(e)}"
    

def _summarize_payloads(query: str, pages_info: list[dict]) -> str:
    """用本地 LLM 将多页文本压缩为聚焦 query 的凝练摘要，大幅节省 VLM token。"""
    if not pages_info:
        return ""

    pages_text = []
    for p in pages_info:
        header = f"--- 文献: {p['doc_id']} | 页码: {p['page_num']} ---"
        pages_text.append(f"{header}\n{p['raw_text'][:3000]}")

    prompt = (
        f"你是一个学术论文分析助手。用户的问题是：'{query}'\n\n"
        f"以下是检索到的论文页面文本。请提取与用户问题直接相关的信息，"
        f"生成一份简洁的摘要（中文，总计不超过500字）：\n\n"
        + "\n\n".join(pages_text)
    )

    try:
        response = requests.post(
            f"{Config.LLM_API_BASE}/chat/completions",
            json={
                "model": Config.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1
            },
            timeout=30
        )
        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content']
            return f"【文本压缩摘要 (聚焦 '{query}')】:\n{summary}"
    except Exception as e:
        print(f"   ⚠️ 文本摘要失败: {e}，降级为截断原文")

    # Fallback: truncated raw text
    return "\n\n".join([
        f"--- 文献: {p['doc_id']} | 页码: {p['page_num']} ---\n{p['raw_text'][:500]}..."
        for p in pages_info
    ])


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def rrf_fusion(vision_hits, text_hits, k=60):
    """倒数排名融合算法 (Reciprocal Rank Fusion)"""
    scores = {}
    hit_dict = {}
    
    # 视觉得分叠加
    for rank, hit in enumerate(vision_hits):
        scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
        hit_dict[hit.id] = hit
        
    # 文本得分叠加
    for rank, hit in enumerate(text_hits):
        scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
        hit_dict[hit.id] = hit
        
    # 根据 RRF 总分降序排列
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [hit_dict[doc_id] for doc_id in sorted_ids]


# ==========================================
# 5. ColPali ROI 智能裁剪：视觉 Token 剪枝
# ==========================================

def _compute_similarity_heatmap(
    query_embeddings: torch.Tensor,
    image_path: str,
    model,
    processor,
    device: str = "cuda"
):
    """
    对单页图片运行 ColPali 前向，计算与 query 的 patch 级相似度热力图。

    Args:
        query_embeddings: ColPali query 输出，shape (1, query_tokens, 128)，在 GPU 上
        image_path: 缓存的 JPEG 页面图片路径
        model: 已加载的 ColPali 模型
        processor: ColPali 处理器
        device: 计算设备

    Returns:
        heatmap: (32, 32) 相似度热力图 tensor
        orig_width: 原始图片宽度 (像素)
        orig_height: 原始图片高度 (像素)
    """
    pil_image = Image.open(image_path)
    orig_width, orig_height = pil_image.size

    batch_images = [pil_image]
    inputs = processor.process_images(batch_images).to(device)

    with torch.no_grad():
        image_embeddings = model(**inputs)

    image_mask = processor.get_image_mask(inputs)
    n_patches = processor.get_n_patches(
        image_size=(448, 448),
        patch_size=14
    )

    similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask
    )
    # similarity_maps: (1, query_tokens, 32, 32)

    heatmap = similarity_maps[0].max(dim=0).values  # (32, 32)

    del inputs, image_embeddings, similarity_maps
    torch.cuda.empty_cache()

    return heatmap, orig_width, orig_height


def _heatmap_to_roi(
    heatmap: torch.Tensor,
    orig_width: int,
    orig_height: int,
    n_patches: tuple = (32, 32),
    top_k: int = 5,
    context_padding: float = 0.25,
    fallback_threshold: float = 0.3
) -> list:
    """
    将 32x32 相似度热力图转换为原始图像上的 ROI 包围盒。

    Args:
        heatmap: (32, 32) 热力图 tensor
        orig_width, orig_height: 原始图片尺寸
        n_patches: ColPali 的 patch 网格 (默认 32, 32)
        top_k: 最多提取的峰值 patch 数量
        context_padding: 上下文外扩比例 (默认 25%)
        fallback_threshold: 若 heatmap.std < threshold * mean → 回退全图

    Returns:
        ROI 列表 [{"x": int, "y": int, "width": int, "height": int, "confidence": float}, ...]
        空列表 = 无显著热点，使用全图
    """
    heatmap_np = heatmap.cpu().float().numpy()
    n_patches_x, n_patches_y = n_patches

    mean_val = heatmap_np.mean()
    std_val = heatmap_np.std()
    if std_val < fallback_threshold * mean_val:
        print("   ⚠️ 相似度分布均匀，无可识别热点区域，回退至全图模式")
        return []

    flat_indices = np.argsort(heatmap_np.ravel())[::-1][:top_k]
    peak_patches = [(int(idx) // n_patches_y, int(idx) % n_patches_y) for idx in flat_indices]

    patch_w = orig_width / n_patches_x
    patch_h = orig_height / n_patches_y

    rois = []
    seen_bounds = set()

    for px, py in peak_patches:
        # px=行(row), py=列(col); 映射回像素: py→x, px→y
        cx = (py + 0.5) * patch_w
        cy = (px + 0.5) * patch_h

        half_w = patch_w * (0.5 + context_padding)
        half_h = patch_h * (0.5 + context_padding)

        x1 = int(max(0, cx - half_w))
        y1 = int(max(0, cy - half_h))
        x2 = int(min(orig_width, cx + half_w))
        y2 = int(min(orig_height, cy + half_h))

        roi_key = (x1 // 50, y1 // 50, x2 // 50, y2 // 50)
        if roi_key not in seen_bounds:
            seen_bounds.add(roi_key)
            rois.append({
                "x": x1, "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(heatmap_np[px, py])
            })

    return rois


def _stitch_rois_to_composite(
    image_path: str,
    rois: list,
    max_images_per_vlm: int = 3
) -> list:
    """
    将多个 ROI 裁剪并编码为 base64。如果 ROI 数量超出 VLM 限制，拼成一张合成图。

    Args:
        image_path: 原始缓存 JPEG 路径
        rois: ROI 包围盒列表
        max_images_per_vlm: VLM 最多接受的图片数 (默认 3)

    Returns:
        base64 编码图片字符串列表 (长度 ≤ max_images_per_vlm)
    """
    pil_image = Image.open(image_path)

    if len(rois) <= max_images_per_vlm:
        # 每个 ROI 单独裁剪编码
        encoded = []
        for roi in rois:
            crop = pil_image.crop((
                roi["x"], roi["y"],
                roi["x"] + roi["width"],
                roi["y"] + roi["height"]
            ))
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=90)
            encoded.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        return encoded

    # 超出限制：将所有 ROI 拼接为一张合成图
    n = len(rois)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # 找出最大的 ROI 宽高作为网格单元尺寸
    max_w = max(r["width"] for r in rois)
    max_h = max(r["height"] for r in rois)

    composite = Image.new("RGB", (cols * max_w, rows * max_h), color=(255, 255, 255))

    for idx, roi in enumerate(rois):
        crop = pil_image.crop((
            roi["x"], roi["y"],
            roi["x"] + roi["width"],
            roi["y"] + roi["height"]
        ))
        # 居中放置在网格单元格中
        row = idx // cols
        col = idx % cols
        paste_x = col * max_w + (max_w - roi["width"]) // 2
        paste_y = row * max_h + (max_h - roi["height"]) // 2
        composite.paste(crop, (paste_x, paste_y))

    buffer = io.BytesIO()
    composite.save(buffer, format="JPEG", quality=90)
    return [base64.b64encode(buffer.getvalue()).decode('utf-8')]


def _crop_and_encode_rois(image_path: str, rois: list) -> list:
    """裁剪 ROI 并编码为 base64 (委托 _stitch_rois_to_composite 处理拼图逻辑)"""
    return _stitch_rois_to_composite(image_path, rois)

class SearchVisionKnowledgeInput(BaseModel):
    query: str = Field(description="必须使用简练的英文学术关键词。用于检索论文的具体内容、细节、公式、架构图、数据集、实验结果等一切内部信息。")
    paper_id: Optional[str] = Field(default=None, description="【极度重要】如果用户是询问刚才提到的、某篇具体的论文，必须传入该论文的 ID（如 2403.02148）。如果是泛泛地提问，则留空。")
@tool("search_vision_knowledge", args_schema=SearchVisionKnowledgeInput)
def search_vision_knowledge(query: str, paper_id: Optional[str] = None) -> str:
    """
    【本地文献深度研读工具】
    当需要回答关于某篇具体论文的内部细节（如方法论、数据集、网络架构、创新点、对比实验等）时，必须调用此工具。
    它会通过多模态引擎阅读本地论文并返回详细的文字分析报告。
    """
    top_k = 3
    print(f"\n🔍 [Agent 后台] 检索视觉记忆库: '{query}' " + (f"| 锁定文献: {paper_id}" if paper_id else "| 全局检索"))

    # --------------------------------------------------
    # 1. 准备双路 Query 向量
    # --------------------------------------------------
    model, processor = _get_vision_model()
    text_model = _get_text_model()
    client = _get_qdrant_client()

    # 视觉路
    inputs = processor.process_queries([query]).to("cuda")
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        query_embeddings = model(**inputs)
    vision_query_vector = query_embeddings[0].cpu().float().numpy().tolist()

    # 文本路
    text_query_vector = text_model.encode(query, normalize_embeddings=True).tolist()

    # 过滤器
    query_filter = None
    if paper_id:
        query_filter = Filter(
            should=[
                FieldCondition(key="paper_id", match=MatchAny(any=[paper_id, paper_id + ".pdf"])),
                FieldCondition(key="file_name", match=MatchAny(any=[paper_id, paper_id + ".pdf"])),
            ]
        )

    # --------------------------------------------------
    # 2. 执行 Qdrant 独立检索 (扩大候选池供 RRF 排序)
    # --------------------------------------------------
    pool_size = top_k * 6 # 扩大候选池，给 RRF 更多素材进行综合排序
    
    vision_res = client.query_points(
        collection_name=collection_name, query=vision_query_vector,
        using="colpali_vision", query_filter=query_filter, limit=pool_size
    ).points
    
    text_res = client.query_points(
        collection_name=collection_name, query=text_query_vector,
        using="text_dense", query_filter=query_filter, limit=pool_size
    ).points

    print(f"🔍 [检索调试] 返回结果数量: {len(vision_res)} (视觉路), {len(text_res)} (文本路)")
    for hit in vision_res:
        print(f"  - Score: {hit.score:.4f}, payload: {hit.payload}")
    for hit in text_res:
        print(f"  - Score: {hit.score:.4f}, payload: {hit.payload}")

    # --------------------------------------------------
    # 3. RRF 融合与截断
    # --------------------------------------------------
    fused_results = rrf_fusion(vision_res, text_res)[:top_k]

    # ---- 评估系统追踪点：捕获双路检索 + RRF 融合的完整中间数据 ----
    try:
        from eval.eval_tracer import EvalContext
        ctx = EvalContext.get()
        if ctx and ctx.enabled:
            ctx.capture_retrieval(
                query=query,
                paper_id=paper_id,
                vision_hits=[{"id": h.id, "score": h.score, "payload": dict(h.payload)} for h in vision_res],
                text_hits=[{"id": h.id, "score": h.score, "payload": dict(h.payload)} for h in text_res],
                fused_hits=[{"id": h.id, "score": h.score, "payload": dict(h.payload)} for h in fused_results],
            )
    except Exception:
        pass  # 评估追踪失败不影响主流程

    if not fused_results: return "未在知识库中找到相关片段。"
    print(f"👁️ RRF 融合成功，交由底层解析图文语义...")
    
    # --------------------------------------------------
    # 4. ROI 智能裁剪 + 组装多模态解析 Payload
    #    利用 ColPali patch 级相似度热力图，裁剪核心公式/表格区域，
    #    以裁剪后的局部图替代全页图发送给 VLM，大幅节省视觉 Token。
    # --------------------------------------------------
    vision_messages = [
        {"type": "text", "text": f"你是一个严谨的学术分析员。用户的问题是：'{query}'。\n"
                                 f"请结合提供的【裁剪后的关键区域图片】，以及下面的【文本摘要】进行综合分析并回答。\n"
                                 f"【护栏规则】：\n"
                                 f"1. 文本摘要中引用的数值、表格数据可作为事实依据，图片用于验证和补充。\n"
                                 f"2. 严禁编造信息，回答时必须说明引用出处。"}
    ]

    pages_info = []
    total_roi_count = 0
    total_fallback_count = 0

    for hit in fused_results:
        doc_id = hit.payload.get('paper_id', hit.payload.get('file_name', '未知文献'))
        page_num = hit.payload.get('page_number', '未知页码')
        img_path = hit.payload.get('image_path', '')
        raw_text = hit.payload.get('page_text', '')
        tables_dict = hit.payload.get('page_tables', {})

        for t_id, md_content in tables_dict.items():
            placeholder = f"[{t_id}_PLACEHOLDER]"
            if placeholder in raw_text:
                raw_text = raw_text.replace(placeholder, f"\n\n📊【表格数据引擎解析】:\n{md_content}\n\n")

        pages_info.append({"doc_id": doc_id, "page_num": page_num, "raw_text": raw_text})

        try:
            if not img_path or not os.path.exists(img_path):
                print(f"   ⚠️ 图片不存在: {img_path}，跳过视觉编码")
                continue

            # ── ROI 智能提取 ──
            try:
                heatmap, orig_w, orig_h = _compute_similarity_heatmap(
                    query_embeddings=query_embeddings,
                    image_path=img_path,
                    model=model,
                    processor=processor,
                    device="cuda"
                )

                rois = _heatmap_to_roi(
                    heatmap=heatmap,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    n_patches=(32, 32),
                    top_k=5,
                    context_padding=0.25
                )

                del heatmap
                torch.cuda.empty_cache()

                if rois:
                    cropped_b64s = _stitch_rois_to_composite(img_path, rois)
                    for b64_img in cropped_b64s:
                        vision_messages.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
                    total_roi_count += len(rois)
                    print(f"   ✂️ 页面 {page_num} 提取 {len(rois)} 个 ROI (拼成 {len(cropped_b64s)} 张图) | 原图 {orig_w}x{orig_h}")
                else:
                    # 无显著热点，回退至全图
                    b64_img = _encode_image_to_base64(img_path)
                    vision_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })
                    total_fallback_count += 1
                    print(f"   ⚠️ 页面 {page_num} 无显著热点，回退至全图模式")

            except Exception as e:
                print(f"   ⚠️ ROI 提取异常 ({e})，回退至全图模式")
                b64_img = _encode_image_to_base64(img_path)
                vision_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
                total_fallback_count += 1

        except Exception as e:
            print(f"   ⚠️ 图片编码失败 {img_path}: {e}")

    # 清理 GPU query embeddings
    del query_embeddings
    torch.cuda.empty_cache()

    if total_roi_count > 0:
        print(f"📐 [Token 剪枝] 共裁剪 {total_roi_count} 个 ROI，{total_fallback_count} 页回退全图，"
              f"预计节省视觉 Token 约 {total_roi_count * 60}%")

    # 用本地 LLM 将多页文本压缩为聚焦 query 的摘要
    condensed_text = _summarize_payloads(query, pages_info)
    vision_messages[0]["text"] += "\n\n" + condensed_text

    # 构建引用来源块
    seen = set()
    source_lines = []
    for p in pages_info:
        key = (p['doc_id'], p['page_num'])
        if key not in seen:
            seen.add(key)
            source_lines.append(f"- 文献: `{p['doc_id']}` | 页码: {p['page_num']} (本地RAG向量库)")
    sources_block = "\n\n---\n📚 **引用来源 (本地RAG)**:\n" + "\n".join(source_lines)

    # 内部请求视觉大模型
    try:
        vl_payload = {
            "model": Config.VLLM_MODEL_NAME,
            "messages": [{"role": "user", "content": vision_messages}],
            "max_tokens": 1024,
            "temperature": 0.3
        }
        vl_response = requests.post(
            f"{Config.VLLM_API_BASE}/chat/completions", json=vl_payload, timeout=60
        )
        if vl_response.status_code != 200:
            print(f"视觉服务解析异常: {vl_response.status_code} - {vl_response.text}，降级返回文本摘要。")
            return condensed_text + sources_block + f"\n\n⚠️ 视觉解析服务返回 {vl_response.status_code}，仅返回文本摘要。"
        vl_data = vl_response.json()
        vision_analysis_text = vl_data['choices'][0]['message']['content']
        
        print(f"👁️ 视觉解析完成，生成综合分析报告。")
        return f"【VLM 双路综合解析报告】:\n{vision_analysis_text}{sources_block}"
    except Exception as e:
        return condensed_text + sources_block + f"\n\n⚠️ 视觉解析服务调用失败 ({e})，仅返回文本摘要。"


if __name__ == "__main__":
    # 测试统一架构：你可以尝试检索一下你目前研究的恶劣天气目标检测方向
    print("\n--- 🧪 V-RAG 统一接口集成测试 ---")
    
    # 模拟步骤 1：寻找最新文献
    # print("1. 尝试全网搜索...")
    # search_res = search_academic_papers("adverse weather object detection fog domain shift", max_results=2)
    # print(search_res)
    
    # 模拟步骤 2：下载并建立视觉索引 (替换为你搜索到的真实 URL)
    # print("\n2. 尝试视觉入库...")
    # ingest_res = download_and_ingest_vision_paper("2403.02148", "https://arxiv.org/pdf/2403.02148.pdf")
    # print(ingest_res)
    pass