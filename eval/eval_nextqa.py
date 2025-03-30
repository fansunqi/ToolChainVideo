import json
import os
from collections import Counter
import argparse
from tqdm import tqdm
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pickle


# 定义选项字母映射
option_map = ["A", "B", "C", "D", "E"]

# LLM for rephrase
chat = ChatOpenAI(
    model="gpt-3.5-turbo",  # 或其他您选择的模型
    temperature=0.0,
    api_key='sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra',
    base_url="https://api.juheai.top/v1"  # OpenAI 的基础 URL
)

# cache
cache_file = "judge_cache.pkl"
use_cache = True
if use_cache and os.path.exists(cache_file):
    print("loading cache...")
    with open(cache_file, "rb") as f:
        llm_cache = pickle.load(f)
else:
    llm_cache = {}
    

def option_character_matching(answer, options):
    
    # 字符匹配（确保只有一个选项字母出现在 answer 中）
    for i, option in enumerate(options):
        if option_map[i] in answer and all(option_map[j] not in answer for j in range(len(options)) if j != i):
            return i, "character matching"
    return -1, "none"

def option_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保只有一个选项完整出现在 answer 中）
    for i, option in enumerate(options):
        if option in answer and all(opt not in answer for j, opt in enumerate(options) if j != i):
            return i, "full matching"
    return -1, "none"

def answer_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保 answer 完整出现在一个选项中）
    for i, option in enumerate(options):
        if answer in option and all(answer not in opt for j, opt in enumerate(options) if j != i):
            return i, "answer full matching"
    return -1, "none"

def LLM_rephrase(answer, options, question):
    
    # TODO 加上 cache
    
    # 首先构造选项的提示
    option_labels = ['A', 'B', 'C', 'D', 'E']
    options_with_labels = "\n".join([f"{label}: {option}" for label, option in zip(option_labels, options)])
    
    # 创建 prompt 给 LLM
    prompt = f"""
    Given the following question and possible answers, determine which option matches the provided answer. 
    Provide only the option letter (A, B, C, D, or E).

    Question: {question}
    Answer: {answer}
    Options:
    {options_with_labels}

    The correct answer option is:
    """
    
    if use_cache and (prompt in llm_cache):
        # 缓存命中
        print("Cache hit!")
        answer_rephrase = llm_cache[prompt]
    else:
        # 缓存未命中
        print("Cache miss. Calling API...")
        
        messages = [HumanMessage(content=prompt)]
        answer_rephrase = chat.invoke(messages).content
        
        # 保存缓存
        llm_cache[prompt] = answer_rephrase
        print("Saving cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(llm_cache, f)
    
    return answer_rephrase
    
    
def get_predicted_option(answer, options):
    """根据答案匹配正确选项"""
    
    predicted_option, match_method = option_character_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    predicted_option, match_method = option_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    predicted_option, match_method = answer_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
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
    error_items = 0

    # 评估每个数据项
    for item in tqdm(data):
        truth = item['truth']
        options = [item['optionA'], item['optionB'], item['optionC'], item['optionD'], item['optionE']]
        question = item['question']
        
        # 说明 answers 是 "Error", 暂时不处理
        if isinstance(item['answers'], str):
            error_items += 1
            continue

        good_answers = item['answers']['good_anwsers']
        
        predicted_options = []
        match_methods = []
        for answer in good_answers:
            predicted_option, match_method = get_predicted_option(answer, options)
            if predicted_option != -1:
                predicted_options.append(predicted_option)
                match_methods.append(match_method)
            else:
                answer_rephrase = LLM_rephrase(answer, options, question)
                predicted_option, match_method = get_predicted_option(answer_rephrase, options)
                predicted_options.append(predicted_option)
                match_methods.append(match_method)   
        
        item["predicted_options"] = predicted_options
        item["match_methods"] = match_methods
        
        # 去除 -1
        predicted_options = [option for option in predicted_options if option != -1]
        
        # 投票确定最终预测答案
        # 多个选项个数一样，随机选一个
        if predicted_options:
            option_counts = Counter(predicted_options)
            most_common_option, _ = option_counts.most_common(1)[0]
            final_predicted_option = most_common_option
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
    print(f"Not Error items: {total_items - error_items}")
    print(f"Have ans items: {have_ans_items}")
    print(f"Correct items: {correct_items}")
    print(f"Acc include no ans: {acc_include_no_ans:.2%}")
    print(f"Acc exclude no ans: {acc_exclude_no_ans:.2%}")

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

