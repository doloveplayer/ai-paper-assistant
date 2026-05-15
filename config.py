import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# =========================================================
# ⚙️ 第一部分：全局配置中心 (Global Configuration)
# 将所有可能变动的路径、URL、参数全部收拢在这里，方便一键修改
# =========================================================
class Config:
    # 1. 存储与网络基建
    QDRANT_STORAGE_URL = "http://localhost:6333"
    COLLECTION_NAME = "cv_papers_collection"
    COLLECTION_VRAG_NAME = "colpali_vision_papers" # 视觉特征的集合名
    COLLECTION_MIX_NAME = "vrag_hybrid_collection" # 视觉特征的集合名
    DATA_DIR = "/home/c2216-3090/disB/hyh/AI/data" # 想学习的 PDF 论文放在这里
    DOWNLOAD_DIR = "/home/c2216-3090/disB/hyh/AI/data/downloads" # 自动下载的论文保存在这
    CACHE_DIR = "/home/c2216-3090/disB/hyh/AI/cache" # 模型缓存目录
    QDRANT_STORAGE_DIR = "/home/c2216-3090/disB/hyh/AI/qdrant_storage" # 向量数据库持久化保存的文件夹
    
    # 2. 模型与算力路径
    EMBEDDING_MODEL_PATH = "/home/c2216-3090/disB/hyh/AI/models/bge-m3/BAAI/bge-m3"
    LLM_API_BASE = "http://localhost:8000/v1"
    LLM_MODEL_NAME = "qwen2.5-7b-instruct"
    VLLM_MODEL_NAME = "qwen2-vl-7b-instruct"
    VLLM_API_BASE = "http://localhost:8001/v1"
    VISION_MODEL_PATH = "/home/c2216-3090/disB/hyh/AI/models/colpali-v1.2"
    
    # 3. RAG 业务超参数
    BEG_EMBEDDING_SIZE = 1024
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200

    # 4. 分层记忆系统超参数
    USER_MEMORY_COLLECTION = "user_memory"       # Qdrant 长期记忆集合
    WORKING_CONTEXT_MAX_ROUNDS = 5                # 工作记忆最大对话轮数
    MEMORY_RETRIEVAL_TOP_K = 3                    # 长期记忆检索返回条数


# 确保下载目录存在
os.makedirs(Config.DOWNLOAD_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)

# 全局初始化 LlamaIndex 设置
# Settings.embed_model = HuggingFaceEmbedding(model_name=Config.EMBEDDING_MODEL_PATH, device="cuda")
# Settings.text_splitter = SentenceSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)

# python -m vllm.entrypoints.openai.api_server \
#     --model ./models/llm/qwen/Qwen2.5-7B-Instruct-AWQ \
#     --served-model-name qwen2.5-7b-instruct \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.35 \
#     --quantization awq \
#     --port 8000 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --enable-prefix-caching

# VLM 视觉模型 (无状态 API，每次请求结束自动释放 KV Cache)
# max-model-len = 文本 + 图片vision_tokens 总和上限，3图约需 6000-8000 tokens
# python -m vllm.entrypoints.openai.api_server \
#     --model ./models/llm/qwen/Qwen2-VL-2B-Instruct-AWQ \
#     --served-model-name qwen2-vl-7b-instruct \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.25 \
#     --limit-mm-per-prompt '{"image": 3}' \
#     --enforce-eager \
#     --port 8001 \
#     --enable-prefix-caching

# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "qwen2-vl-7b-instruct",
#     "messages": [{"role": "user", "content": "搜索关于目标检测的论文"}],
#     "tools": [{
#       "type": "function",
#       "function": {
#         "name": "search_academic_papers",
#         "description": "搜索论文",
#         "parameters": {
#           "type": "object",
#           "properties": {
#             "primary_concept": {"type": "string"},
#             "secondary_concept": {"type": "string"}
#           }
#         }
#       }
#     }],
#     "tool_choice": "auto"
#   }'

# python -m chainlit run app_ui.py --port 8053

# 运行docker指令 -d 指定模式为后台运行，-p 指定端口映射，-v 指定数据卷映射（本地目录映射到容器内） :z确保 Docker 有权读写这个本地目录 qdrant/qdrant: 指定镜像名称。
# 注意：如果你之前已经运行过一次这个命令，第二次运行会提示端口被占用，这时你可以先停止之前的容器（docker ps -a 查看容器ID，docker stop <容器ID> 停止容器），或者换一个端口映射（比如 -p 6335:6333）来避免冲突。
# sudo docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage:z qdrant/qdrant
