import os
import qdrant_client
from config import Config
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

# 1. 基础配置：设定你的模型和数据路径
DATA_DIR = Config.DATA_DIR
QDRANT_STORAGE_URL = Config.QDRANT_STORAGE_URL
EMBEDDING_MODEL_PATH = Config.EMBEDDING_MODEL_PATH
QDRANT_STORAGE_DIR = Config.QDRANT_STORAGE_DIR
COLLECTION_NAME = Config.COLLECTION_NAME

# 2. 全局设置 Embedding 模型
# 使用 HuggingFaceEmbedding 直接加载本地模型，跑在 GPU 上
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_PATH,
    device="cuda" # 确保你的 3090Ti 正在轰鸣
)

# 注意：在纯“入库”阶段，我们不需要配置大语言模型 (LLM)，所以暂时将其置空或使用默认
Settings.llm = None 

# 数据切块策略 (Chunking)
# 对于 SCI 论文，句子结构的完整性极其重要，切忌从中间暴力截断句子。
# 我们设定每块 512 个 Token，块与块之间重叠 50 个 Token 以防上下文丢失。
Settings.text_splitter = SentenceSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)

def ingest_papers_to_qdrant():
    # 未来如果有其他格式，可以在字典里增加，如 ".docx": DocxReader()
    file_extractor = {".pdf": PyMuPDFReader()}

    print(f"正在从 {DATA_DIR} 读取文档...")
    # 3. 数据加载与解析 (Load)
    # SimpleDirectoryReader 会自动识别 PDF、TXT 等格式并提取文字
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        file_extractor=file_extractor,
        recursive=False # 如果需要读子目录改 True
    )

    documents = reader.load_data(show_progress=True)

    if not documents:
        print("未发现有效文档。")
        return

    print("\n" + "="*50)
    print(f"📋 预览模式：已读取 {len(documents)} 个文档片段。")
    print("展示前 3 个片段的解析结果（前 300 字）：")

    for i, doc in enumerate(documents[:3]):
        print(f"\n[文件 {i+1}: {doc.metadata.get('file_name', '未知')}]")
        # 打印清理后的文本预览
        content_preview = doc.text[:300].replace('\n', ' ')
        print(f"内容摘要: {content_preview}...")

    confirm = input("\n⚠️ 解析结果是否存在乱码？确认入库 (y) / 终止任务 (n): ")
    if confirm.lower() != 'y':
        print("任务已终止。请检查 PDF 格式或尝试 OCR。")
        return

    client = qdrant_client.QdrantClient(url=QDRANT_STORAGE_URL)
    if client.collection_exists(COLLECTION_NAME):
        # client.delete_collection(COLLECTION_NAME) # 视情况取消注释：每次运行都清空
        print(f"集合 {COLLECTION_NAME} 已存在，正在追加数据...")
    
    # 定义这个知识库的集合名称
    vector_store = QdrantVectorStore(client=client, collection_name="cv_papers_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("正在进行文本切块与特征向量化计算，并存入 Qdrant (这可能会占用一定的 GPU 算力)...")
    
    # 6. 核心动作：执行全链路入库
    # 从文档 -> 切块 -> 调用 BGE-m3 转成向量 -> 写入 Qdrant 数据库
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"入库完成！向量数据已持久化保存在 {QDRANT_STORAGE_DIR} 目录。")

if __name__ == "__main__":
    # 确保 data 文件夹存在，你可以在里面扔几篇目标检测方向的 PDF
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(QDRANT_STORAGE_DIR, exist_ok=True)
    
    # 检查是否有文件，没有则提示
    if not os.listdir(DATA_DIR):
        print(f"请先往 {DATA_DIR} 文件夹中放入一些 PDF 测试文件！")
    else:
        ingest_papers_to_qdrant()