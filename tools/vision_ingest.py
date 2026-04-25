import sys
from pathlib import Path
# 将项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import uuid
import torch
from pdf2image import convert_from_path
import qdrant_client
from qdrant_client.models import Distance, VectorParams, MultiVectorConfig, PointStruct, MultiVectorComparator
from qdrant_client.models import Filter, FieldCondition, MatchValue

import transformers
try:
    import transformers.integrations.peft
    # 直接将底层有 Bug 的函数替换为一个匿名函数（什么都不做，原样返回 config）
    transformers.integrations.peft._convert_peft_config_moe = lambda peft_config, model_type: peft_config
    print("🔧 [系统补丁] 已成功拦截并废除 transformers 的 MoE 字符串转换 Bug！")
except Exception as e:
    pass

from colpali_engine.models import ColPali, ColPaliProcessor
from config import Config

# 1. 基础配置
DATA_DIR = Config.DATA_DIR
QDRANT_STORAGE_URL = Config.QDRANT_STORAGE_URL
COLLECTION_NAME = Config.COLLECTION_VRAG_NAME  # 为视觉向量单独建一个新集合

def ingest_papers_to_qdrant_vision():
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
    # 第二阶段：初始化视觉大模型与 Qdrant 多向量集合
    # ---------------------------------------------------------
    print("\n⏳ 正在加载 ColPali 视觉向量模型到 GPU...")
    # 将这里改成你刚才下载的本地文件夹路径！
    local_model_path = "/home/c2216-3090/disB/hyh/AI/models/colpali-v1.2" 
    
    # 代码完全不用变，它会自动识别这是一个本地路径
    model = ColPali.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map="cuda",local_files_only=True)
    processor = ColPaliProcessor.from_pretrained(local_model_path, local_files_only=True)
    model.eval() # 开启推理模式

    print("🔌 正在连接 Qdrant 向量数据库...")
    client = qdrant_client.QdrantClient(url=QDRANT_STORAGE_URL)
    
    # 🚨 核心：必须开启 MultiVectorConfig 才能支持 ColPali 的多向量矩阵！
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=128, # ColPali 每一个图像 Patch 的特征维度是 128
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ) 
            )
        )
        print(f"✅ 创建了全新的多向量集合: {COLLECTION_NAME}")
    else:
        print(f"✅ 集合 {COLLECTION_NAME} 已存在，正在追加数据...")

    # ---------------------------------------------------------
    # 第三阶段：防 OOM 批处理入库
    # ---------------------------------------------------------
    # 考虑到 3090Ti 的显存，我们每次只扔 4 张图片(页) 给 ColPali，防止显存爆掉
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
        # 将整本 PDF 转化为 PIL Image 列表
        page_images = convert_from_path(pdf_path)
        total_pages = len(page_images)

        # ---------------------------------------------------------
        # ⚡ 3. 核心断点续传路由判断
        # ---------------------------------------------------------
        if ingested_count >= total_pages:
            print(f"⏭️  检测到 {file_name} 已全部入库 ({total_pages}/{total_pages} 页)，完美跳过...")
            continue
        elif ingested_count > 0:
            print(f"⚡ 触发断点续传：已存在 {ingested_count} 页，将从第 {ingested_count + 1} 页开始继续提取...")
        else:
            print(f"   => 发现新文件，共 {total_pages} 页，开始进行多模态特征编码...")

        # ---------------------------------------------------------
        # ⚡ 4. 循环起点不再是 0，而是从 ingested_count 开始
        # ---------------------------------------------------------
        # 例如：之前存了 4 页，ingested_count=4。循环就从索引 4 (即第 5 页) 开始，完美衔接！
        for i in range(ingested_count, total_pages, BATCH_SIZE):
            points = []

            batch_images = page_images[i : i + BATCH_SIZE]
            
            # 使用 ColPali 处理器对图片进行编码
            inputs = processor.process_images(batch_images).to("cuda")
            
            with torch.no_grad():
                # 输出的 image_embeddings 维度: [batch_size, num_patches, 128]
                image_embeddings = model(**inputs)
            
            # 组装 Qdrant 所需的 PointStruct
            for j, embedding in enumerate(image_embeddings):
                page_num = i + j + 1
                point_id = uuid.uuid4().hex

                # 将图片缓存到本地硬盘
                img_save_path = os.path.abspath(f"image_cache/{file_name}_page_{page_num}.jpg")
                batch_images[j].save(img_save_path, "JPEG")
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.cpu().float().numpy().tolist(), 
                        payload={
                            "file_name": file_name,
                            "page_number": page_num,
                            "image_path": img_save_path
                        }
                    )
                )
                
            # 分批入库
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   ⚙️ 处理并入库进度: {min(i + BATCH_SIZE, total_pages)} / {total_pages} 页")

        print(f"✅ 文件 {file_name} 已处理完毕！")

    print("\n🎉 全部入库完成！多模态视觉数据已持久化保存。")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("/home/c2216-3090/disB/hyh/AI/image_cache", exist_ok=True)
    ingest_papers_to_qdrant_vision()