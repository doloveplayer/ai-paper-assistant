from modelscope import snapshot_download

print("🚀 开始下载 Qwen2.5-7B-Instruct (约需 15GB 硬盘空间)...")

# 下载模型权重
model_dir = snapshot_download(
    'qwen/Qwen2-VL-2B-Instruct-AWQ', 
    cache_dir='./models/llm'
)

print(f"✅ LLM 下载完成！模型保存在: {model_dir}")

# import torch
# print(torch.cuda.is_available())  # 如果输出 True，恭喜，病灶彻底切除！

# python -c "import torch; print('CUDA 可用状态:', torch.cuda.is_available(), '| 识别到的显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"