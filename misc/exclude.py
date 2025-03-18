import os

def get_all_subdirectories(path):
    subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return subdirectories

# 示例路径
path = "/hf_home/hub/spaces/next-qa/NExTVideo"

# 获取所有子文件夹名称
subdirectories = get_all_subdirectories(path)
print(len(subdirectories))