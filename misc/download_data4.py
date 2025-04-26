from huggingface_hub import login, hf_hub_download
import os

# 登录 Hugging Face（如果需要）
# login(token="your_huggingface_token")

# 数据集仓库信息
repo_id = "lmms-lab/Video-MME"
repo_type = "dataset"

# 指定下载路径
download_dir = "/mnt/Shared_03/fsq/VideoMME"  # 替换为你的目标路径
os.makedirs(download_dir, exist_ok=True)

# 下载所有文件
for index in range(1, 21):
    if index < 10:
        filename = f"videos_chunked_0{index}.zip"
    else:   
        filename = f"videos_chunked_{index}.zip"
    print(f"Downloading {filename}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=download_dir,  # 指定下载路径
        local_dir_use_symlinks=False  # 直接下载文件而不是创建符号链接
    )
    print(f"Downloaded {filename} to: {file_path}")