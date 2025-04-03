import json
from difflib import SequenceMatcher
from transformers import pipeline

# 加载 JSON 文件
with open('/home/fsq/ToolChainVideo/output/nextqa/nextqa_output_20250318_172238.json', 'r') as f:
    data = json.load(f)

# 初始化 LLM 判断模型
llm = pipeline("text-classification", model="facebook/bart-large-mnli")

def is_good_answer(truth, answer):
    # 字符匹配
    if truth.lower() in answer.lower():
        return True
    
    # 语义匹配
    similarity = SequenceMatcher(None, truth.lower(), answer.lower()).ratio()
    if similarity > 0.8:
        return True
    
    # LLM 判断
    result = llm(f"Is the answer '{answer}' correct for the question with truth '{truth}'?")
    if result[0]['label'] == 'ENTAILMENT':
        return True
    
    return False

total_items = len(data)
correct_items = 0

# 评估每个数据项
for item in data:
    truth = item['truth']
    good_answers = item['answers']['good_anwsers']

    # 定义选项字母映射
    option_map = ["A", "B", "C", "D", "E"]
    # 根据 truth 找到对应的选项
    truth_option = option_map[truth]


    is_correct = any(is_good_answer(truth, answer) for answer in good_answers)
    
    if is_correct:
        correct_items += 1
    
    item['is_correct'] = is_correct

accuracy = correct_items / total_items

# 输出结果
print(f"Total items: {total_items}")
print(f"Accuracy: {accuracy:.2%}")

for item in data:
    print(f"QID: {item['qid']}, Correct: {item['is_correct']}")

# 保存评估结果到文件
with open('/home/fsq/ToolChainVideo/output/nextqa/nextqa_output_20250318_172238_evaluated.json', 'w') as f:
    json.dump(data, f, indent=4)