import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import re
import uuid
import torch
import fitz  # 引入 PyMuPDF
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer # 引入文本向量库

import qdrant_client
from qdrant_client.models import Distance, VectorParams, MultiVectorConfig, PointStruct, MultiVectorComparator
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Monkey Patch: 绕过 transformers PEFT _convert_peft_config_moe 对 PaliGemma 的 KeyError
import transformers
try:
    import transformers.integrations.peft
    transformers.integrations.peft._convert_peft_config_moe = lambda peft_config, model_type: peft_config
except Exception:
    pass

from colpali_engine.models import ColPali, ColPaliProcessor
from config import Config

# 1. 基础配置
DATA_DIR = Config.DATA_DIR
CACHE_DIR = Config.CACHE_DIR
QDRANT_STORAGE_URL = Config.QDRANT_STORAGE_URL
COLLECTION_NAME = Config.COLLECTION_MIX_NAME

# ==========================================
# 🛠️ 文本清洗器：滤除 PyMuPDF 乱码
# ==========================================
def extract_text_and_tables_from_page(page):
    """
    高级文档解析器：分离文本与表格，在文本中插入占位符。
    返回: (clean_text, tables_dict)
    """
    # 1. 寻找页面上的所有表格
    tables = page.find_tables()
    
    tables_payload = {}
    table_rects = []
    
    # 2. 将表格转化为 Markdown，并记录它们的坐标
    if tables.tables:
        for i, tab in enumerate(tables.tables):
            # 将表格转换为 pandas DataFrame
            try:
                df = tab.to_pandas()
                # 清洗掉可能为空的列名
                df.columns = [str(c) if c else f"Col_{j}" for j, c in enumerate(df.columns)]
                markdown_table = df.to_markdown(index=False)
                
                table_id = f"TABLE_{i}"
                tables_payload[table_id] = markdown_table
                table_rects.append((table_id, fitz.Rect(tab.bbox)))
            except Exception as e:
                print(f"表格解析警告: {e}")
                continue

    # 3. 按区块 (blocks) 获取页面文本，包含排版坐标
    blocks = page.get_text("blocks")
    # 按 Y 坐标从上到下排序，保证阅读顺序
    blocks.sort(key=lambda b: b[1])
    
    final_text_parts = []
    processed_tables = set()

    # 4. 遍历文本块，进行“空间碰撞检测”
    for b in blocks:
        block_rect = fitz.Rect(b[:4])
        block_text = b[4]
        
        is_inside_table = False
        
        for table_id, t_rect in table_rects:
            # 如果文本块与表格坐标有重合（Intersect）
            if block_rect.intersects(t_rect):
                is_inside_table = True
                # 只有第一次碰到这个表格时，才插入“眼”（占位符）
                if table_id not in processed_tables:
                    final_text_parts.append(f"\n\n[{table_id}_PLACEHOLDER]\n\n")
                    processed_tables.add(table_id)
                break # 跳出当前表格判断
                
        # 如果这个文本块不在任何表格内，进行常规清洗后加入正文
        if not is_inside_table:
            # 常规过滤乱码
            clean_block = re.sub(r'([0-9a-fA-F]{10,})', '', block_text)
            clean_block = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', clean_block)
            clean_block = re.sub(r'[ \t]{2,}', ' ', clean_block).strip()
            if clean_block:
                final_text_parts.append(clean_block)

    # 组装最终带占位符的干净文本
    page_text = "\n".join(final_text_parts)
    
    return page_text, tables_payload

def ingest_papers_to_qdrant_hybrid():
    # ---------------------------------------------------------
    # 第一阶段：扫描本地目录并生成预览
    # ---------------------------------------------------------
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️ 未在 {DATA_DIR} 发现 PDF 文件。")
        return

    print("\n" + "="*50)
    print(f"📋 预览模式：已发现 {len(pdf_files)} 个 PDF 文档准备进行视觉特征提取。")
    print("展示待处理文件列表：")
    for i, file_name in enumerate(pdf_files):
        # 简单打印一下文件大小
        file_size_mb = os.path.getsize(os.path.join(DATA_DIR, file_name)) / (1024 * 1024)
        print(f"  [{i+1}] {file_name} ({file_size_mb:.2f} MB)")

    confirm = input("\n⚠️ 视觉模型入库极其消耗显存，请确保已关闭 vLLM 服务！\n确认开始视觉入库 (y) / 终止任务 (n): ")
    if confirm.lower() != 'y':
        print("任务已终止。")
        return

    # ---------------------------------------------------------
    # 初始化：双路模型加载 (GPU 视觉 + CPU 文本)
    # ---------------------------------------------------------
    print("\n⏳ 正在加载 ColPali 视觉大模型 (GPU)...")
    local_model_path = "/home/c2216-3090/disB/hyh/AI/models/colpali-v1.2" 
    model = ColPali.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map="cuda",local_files_only=True)
    processor = ColPaliProcessor.from_pretrained(local_model_path, local_files_only=True)
    model.eval()

    print("⏳ 正在加载 BGE 文本向量模型 (CPU)...")
    # 强制在 CPU 上运行，BGE-M3 跑单页文本在 CPU 上仅需几十毫秒
    text_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")

    print("🔌 正在连接 Qdrant 向量数据库...")
    client = qdrant_client.QdrantClient(url=QDRANT_STORAGE_URL)
    
    # 🚨 核心：定义双轨命名向量 (Named Vectors)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                # 视觉通道：多维矩阵
                "colpali_vision": VectorParams(
                    size=128, 
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)
                ),
                # 文本通道：稠密单向量
                "text_dense": VectorParams(
                    size=1024, 
                    distance=Distance.COSINE
                )
            }
        )
        print(f"✅ 创建了全新的【双路混合】多向量集合: {COLLECTION_NAME}")

    BATCH_SIZE = 4 

    for file_name in pdf_files:
        # ---------------------------------------------------------
        # 1. 查询数据库，获取该文件【已经存了多少页】
        # ---------------------------------------------------------
        ingested_count = 0
        try:
            exact_match_filter = Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
            )
            count_result = client.count(collection_name=COLLECTION_NAME, count_filter=exact_match_filter)
            ingested_count = count_result.count
        except Exception as e:
            pass # 如果是第一次建表，可能会报错，忽略即可 (默认 0)

        # ---------------------------------------------------------
        # 2. 渲染 PDF 并获取总页数
        # ---------------------------------------------------------
        pdf_path = os.path.join(DATA_DIR, file_name)
        print(f"\n📄 正在读取 {file_name} ...")
        
        # 同时打开图片流和 PyMuPDF 文本流
        page_images = convert_from_path(pdf_path)
        pdf_doc = fitz.open(pdf_path) 
        total_pages = len(page_images)

        for i in range(ingested_count, total_pages, BATCH_SIZE):
            points = []
            batch_images = page_images[i : i + BATCH_SIZE]
            
            # [视觉路] 提取图像 Embedding
            inputs = processor.process_images(batch_images).to("cuda")
            with torch.no_grad():
                image_embeddings = model(**inputs)
            
            for j, vision_embedding in enumerate(image_embeddings):
                page_num = i + j + 1
                point_id = uuid.uuid4().hex
                
                # [文本路] 提取并清洗文本
                page_text, page_tables = extract_text_and_tables_from_page(pdf_doc[page_num - 1])
                
                # 如果当前页没有提取到有效文本，赋予一个全零向量或默认提示向量
                if page_text:
                    text_embedding = text_model.encode(page_text, normalize_embeddings=True)
                else:
                    text_embedding = [0.0] * 1024
                    page_text = "[当前页面以图表为主，未提取到有效文本]"

                img_save_path = os.path.abspath(f"image_cache/{file_name}_page_{page_num}.jpg")
                batch_images[j].save(img_save_path, "JPEG")
                
                # 🚨 核心组合：将两路向量和文本一起打包进 PointStruct
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "colpali_vision": vision_embedding.cpu().float().numpy().tolist(),
                            "text_dense": text_embedding.tolist()
                        },
                        payload={
                            "file_name": file_name,
                            "page_number": page_num,
                            "image_path": img_save_path,
                            "page_text": page_text,  # <--- 文本 Payload 挂载
                            "page_tables": page_tables  # <--- 表格 Payload 挂载
                        }
                    )
                )
                
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   ⚙️ 双路映射进度: {min(i + BATCH_SIZE, total_pages)} / {total_pages} 页")

        pdf_doc.close()
        print(f"✅ 文件 {file_name} 已处理完毕！")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    ingest_papers_to_qdrant_hybrid()