import json
from difflib import SequenceMatcher
from transformers import pipeline
from collections import Counter

# 加载 JSON 文件
with open('/home/fsq/ToolChainVideo/output/nextqa/nextqa_output_20250318_172238.json', 'r') as f:
    data = json.load(f)

# 初始化 LLM 判断模型
llm = pipeline("text-classification", model="facebook/bart-large-mnli")

# 定义选项字母映射
option_map = ["A", "B", "C", "D", "E"]

def get_predicted_option(answer, options):
    # 字符匹配
    for i, option in enumerate(options):
        if option_map[i] in answer:
            return i, "character matching"
    
    # 语义匹配
    for i, option in enumerate(options):
        similarity = SequenceMatcher(None, option.lower(), answer.lower()).ratio()
        if similarity > 0.8:
            return i, "semantic matching"
    
    # LLM 判断
    for i, option in enumerate(options):
        result = llm(f"Is the answer '{answer}' correct for the option '{option}'?")
        if result[0]['label'] == 'ENTAILMENT':
            return i, "LLM matching"
    
    return None, "none"

total_items = len(data)
correct_items = 0

# 评估每个数据项
for item in data:
    truth = item['truth']
    options = [item['optionA'], item['optionB'], item['optionC'], item['optionD'], item['optionE']]
    good_answers = item['answers']['good_anwsers']
    
    predicted_options = []
    match_methods = []
    for answer in good_answers:
        predicted_option, match_method = get_predicted_option(answer, options)
        if predicted_option is not None:
            predicted_options.append(predicted_option)
            match_methods.append(match_method)
    
    # 投票确定最终预测答案
    if predicted_options:
        option_counts = Counter(predicted_options)
        most_common_option, _ = option_counts.most_common(1)[0]
        final_predicted_option = most_common_option
    else:
        final_predicted_option = None
    
    is_correct = (final_predicted_option == truth)
    
    if is_correct:
        correct_items += 1
    
    item['predicted_option'] = option_map[final_predicted_option] if final_predicted_option is not None else None
    item['is_correct'] = is_correct
    item['match_methods'] = match_methods

accuracy = correct_items / total_items

# 输出结果
print(f"Total items: {total_items}")
print(f"Accuracy: {accuracy:.2%}")

for item in data:
    print(f"QID: {item['qid']}, Predicted Option: {item['predicted_option']}, Correct: {item['is_correct']}, Match Methods: {item['match_methods']}")

# 保存评估结果到文件
with open('/home/fsq/ToolChainVideo/output/nextqa/nextqa_output_20250318_172238_evaluated.json', 'w') as f:
    json.dump(data, f, indent=4)