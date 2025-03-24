import json
import os
from difflib import SequenceMatcher
from transformers import pipeline
from collections import Counter
import argparse
from tqdm import tqdm

# 定义选项字母映射
option_map = ["A", "B", "C", "D", "E"]
llm = None  # LLM 先不初始化

def load_llm():
    """懒加载 LLM，只在需要时才初始化"""
    global llm
    if llm is None:
        print("loading LLM to judge...")
        llm = pipeline("text-classification", model="facebook/bart-large-mnli")
        print("LLM loaded.")

def get_predicted_option(answer, options):
    """根据答案匹配正确选项"""
    # 字符匹配
    # 字符匹配（确保只有一个选项字母出现在 answer 中）
    for i, option in enumerate(options):
        if option_map[i] in answer and all(option_map[j] not in answer for j in range(len(options)) if j != i):
            return i, "character matching"
    
    # 语义匹配 最长公共子序列
    # 计算所有选项的相似度
    similarities = [SequenceMatcher(None, option.lower(), answer.lower()).ratio() for option in options]
    # 找到相似度大于 0.8 的选项索引
    high_similarity_indices = [i for i, sim in enumerate(similarities) if sim > 0.8]
    # 只有一个选项的相似度大于 0.8，才返回
    if len(high_similarity_indices) == 1:
        return high_similarity_indices[0], "semantic matching"
    
    # LLM 判断
    load_llm()  # 只有在需要 LLM 判断时才加载
    entailment_indices = []
    for i, option in enumerate(options):
        result = llm(f"Is the answer '{answer}' correct for the option '{option}'?")
        if result[0]['label'] == 'ENTAILMENT':
            entailment_indices.append(i)
    # 只有一个选项是 'ENTAILMENT'，才返回
    if len(entailment_indices) == 1:
        return entailment_indices[0], "LLM matching"
    
    return -1, "none"

def get_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    latest_file = sorted(files)[-1] 
    return latest_file

def main(input_file, output_file):
    # 加载 JSON 文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_items = len(data)
    have_ans_items = 0
    correct_items = 0

    # 评估每个数据项
    for item in tqdm(data):
        truth = item['truth']
        options = [item['optionA'], item['optionB'], item['optionC'], item['optionD'], item['optionE']]

        # 说明 answers 是 None, 暂时不处理
        if isinstance(item['answers'], str):
            continue

        good_answers = item['answers']['good_anwsers']
        
        predicted_options = []
        match_methods = []
        for answer in good_answers:
            predicted_option, match_method = get_predicted_option(answer, options)
            if predicted_option is not None:
                predicted_options.append(predicted_option)
                match_methods.append(match_method)
        
        item["predicted_options"] = predicted_options
        item["match_methods"] = match_methods
        
        # 去除 -1
        predicted_options = [option for option in predicted_options if option != -1]
        
        # 投票确定最终预测答案
        # 多个选项个数一样，随机选一个
        # if predicted_options:
        #     option_counts = Counter(predicted_options)
        #     most_common_option, _ = option_counts.most_common(1)[0]
        #     final_predicted_option = most_common_option
        #     have_ans_items += 1
        # else:
        #     final_predicted_option = None
        
        # 投票确定最终预测答案
        if predicted_options:
            option_counts = Counter(predicted_options)
            most_common_options = option_counts.most_common()
            max_count = most_common_options[0][1]
            # 找到所有出现次数等于 max_count 的选项
            candidates = [option for option, count in most_common_options if count == max_count]
            # 选择索引最大的选项
            final_predicted_option = max(candidates)
            have_ans_items += 1
        else:
            final_predicted_option = None
        
        is_correct = (final_predicted_option == truth)
        
        if is_correct:
            correct_items += 1
        
        item['final_predicted_option'] = final_predicted_option if final_predicted_option is not None else None
        item['is_correct'] = is_correct
        item['match_methods'] = match_methods

    acc_include_no_ans = correct_items / total_items
    acc_exclude_no_ans = correct_items / have_ans_items

    # 输出结果
    print(f"Total items: {total_items}")
    print(f"Have ans items: {have_ans_items}")
    print(f"Correct items: {correct_items}")
    print(f"Acc include no ans: {acc_include_no_ans:.2%}")
    print(f"Acc exclude no ans: {acc_exclude_no_ans:.2%}")

    # for item in data:
    #     print(f"QID: {item['qid']}, Predicted Option: {item['predicted_option']}, Correct: {item['is_correct']}, Match Methods: {item['match_methods']}")

    # 保存评估结果到文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NextQA answers")
    parser.add_argument('--input_file', type=str, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, help="Path to the output JSON file")
    args = parser.parse_args()

    if not args.input_file:
        args.input_file = get_latest_file('output/nextqa')
    
    if not args.output_file:
        # args.output_file = args.input_file.replace('.json', '_eval.json')
        args.output_file = args.input_file.replace('output/nextqa', 'eval/nextqa')

    main(args.input_file, args.output_file)



'''
只有字符匹配：
Total items: 100
Have ans items: 61
Correct items: 27
Acc include no ans: 27.00%
Acc exclude no ans: 44.26%
'''
# 加了下面两种方式之后还是这样
