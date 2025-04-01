import os

path = '/share_data/NExT-QA/NExTVideo/1002/13205297255.mp4'

# 获取文件名（包括扩展名）
filename_with_ext = os.path.basename(path)

# 去掉扩展名
filename = os.path.splitext(filename_with_ext)[0]

print(filename)  # 输出: 13205297255