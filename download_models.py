from modelscope import snapshot_download

# 下载 BGE-m3 向量模型
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='./models/bge-m3')
print(f"Embedding模型已下载至: {model_dir}")