import diskcache as dc
from tqdm import tqdm

# 假设 cache 路径跟你的代码一样
cache_path = '/home/fsq/.cache/octotools/cache_openai_gpt-4o.db'  # 换成你的真实路径
cache = dc.Cache(cache_path)

# 遍历所有 keys，查找包含 "ACAD" 的项
target = '''I will show you an image sequence of 16 sampled frames from a video. I have annotated the images with numbered circles. Based on the image sequence, try to answer this question: 
Which of the following players serves from the backcourt? Choose your answer from below options: A.Forward, B.Goalkeeper, C.Midfielder, D.Defender.'''
for key in tqdm(cache.iterkeys()):
    if target in key:
        print(f"Found key: {key}")
        print(f"Value: {cache[key]}")
        print("-----------------------------")
