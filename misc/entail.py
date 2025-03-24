from difflib import SequenceMatcher

options = [
    "watching the other car",
    "stays still",
    "look at police car",
    "change lane",
    "watch jeep",
]

answer = "The white car in front stays still."


def load_llm():
    """懒加载 LLM，只在需要时才初始化"""
    global llm
    if llm is None:
        print("loading LLM to judge...")
        llm = pipeline("text-classification", model="facebook/bart-large-mnli")
        print("LLM loaded.")


# 语义匹配 最长公共子序列
# 计算所有选项的相似度
similarities = [SequenceMatcher(None, option.lower(), answer.lower()).ratio() for option in options]
# 找到相似度大于 0.8 的选项索引
print(similarities)
high_similarity_indices = [i for i, sim in enumerate(similarities) if sim > 0.8]
# 只有一个选项的相似度大于 0.8，才返回
if len(high_similarity_indices) == 1:
    # return high_similarity_indices[0], "semantic matching"
    print(high_similarity_indices[0], "semantic matching")

'''
# LLM 判断
load_llm()  # 只有在需要 LLM 判断时才加载
entailment_indices = []
for i, option in enumerate(options):
    result = llm(f"Is the answer '{answer}' correct for the option '{option}'?")
    if result[0]['label'] == 'ENTAILMENT':
        entailment_indices.append(i)
# 只有一个选项是 'ENTAILMENT'，才返回
if len(entailment_indices) == 1:
    print(entailment_indices[0], "LLM matching")
    # return entailment_indices[0], "LLM matching"
'''
