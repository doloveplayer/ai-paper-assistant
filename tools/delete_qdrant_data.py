import qdrant_client
client = qdrant_client.QdrantClient(url="http://localhost:6333")
# 彻底删除旧集合
client.delete_collection(collection_name="colpali_vision_papers")
print("旧集合已清理。")