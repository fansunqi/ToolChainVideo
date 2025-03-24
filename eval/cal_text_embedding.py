from sentence_transformers import SentenceTransformer
import numpy as np

# 选择一个开源 embedding 模型（可以换成别的更适合的）
model = SentenceTransformer("all-MiniLM-L6-v2")  # 轻量高效，适用于大部分任务

options = [
    "watching the other car",
    "stays still",
    "look at police car",
    "change lane",
    "watch jeep",
]
answer = "The white car in front stays still."

# 计算嵌入向量
option_embeddings = model.encode(options)
answer_embedding = model.encode(answer)

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = [cosine_similarity(answer_embedding, opt_emb) for opt_emb in option_embeddings]
print(similarities)

# 找到最匹配的选项
best_match_index = np.argmax(similarities)
best_match = options[best_match_index]

print("Best matching option:", best_match)
