import os
import pdb
import json
import pickle
import argparse
from tqdm import tqdm
from collections import Counter
from omegaconf import OmegaConf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def option_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保只有一个选项完整出现在 answer 中）
    for i, option in enumerate(options):
        if option in answer and all(opt not in answer for j, opt in enumerate(options) if j != i):
            return i, "option full matching"
    return -1, "none"

def answer_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保 answer 完整出现在一个选项中）
    for i, option in enumerate(options):
        if answer in option and all(answer not in opt for j, opt in enumerate(options) if j != i):
            return i, "answer full matching"
    return -1, "none"

def LLM_rephrase(answer, options, question, conf, eval_llm, llm_cache):
    
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
    
    if conf.eval.use_cache and (prompt in llm_cache):
        # 缓存命中
        print("Cache hit!")
        answer_rephrase = llm_cache[prompt]
    else:
        # 缓存未命中
        print("Cache miss. Calling API...")
        
        messages = [HumanMessage(content=prompt)]
        answer_rephrase = eval_llm.invoke(messages).content
        
        # 保存缓存
        llm_cache[prompt] = answer_rephrase
        print("Saving cache...")
        with open(conf.eval.eval_cache_file, "wb") as f:
            pickle.dump(llm_cache, f)
    
    return answer_rephrase
    
    
def get_predicted_option(answer, options):
    """根据答案匹配正确选项"""
    
    predicted_option, match_method = option_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    predicted_option, match_method = answer_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    return -1, "none"


def get_predicted_option_with_rephrase(answer, options, question, conf, eval_llm, llm_cache):
    predicted_option, match_method = get_predicted_option(answer, options)
    if predicted_option == -1:
        answer_rephrase = LLM_rephrase(answer, options, question, conf, eval_llm, llm_cache)
        predicted_option, match_method = get_predicted_option(answer_rephrase, options)
    return predicted_option, match_method


def get_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if 
             (os.path.isfile(os.path.join(directory, f))
              and f.endswith('.json'))]
    latest_file = sorted(files)[-1] 
    return latest_file


def main(data, output_file, conf, eval_llm, llm_cache):

    total_items = len(data)
    have_ans_items = 0
    correct_items = 0
    error_items = 0

    for item in tqdm(data):
        truth = item['truth']
        options = [item['optionA'], item['optionB'], item['optionC'], item['optionD'], item['optionE']]
        question = item['question']
        
        if not isinstance(item['answers'], list):
            error_items += 1
            continue
        
        if all(answer == "Error" for answer in item['answers']):
            error_items += 1
            continue

        answers = item['answers']
        
        predicted_options = []
        match_methods = []
        for answer in answers:
            predicted_option, match_method = get_predicted_option_with_rephrase(
                answer, options, question, conf, eval_llm, llm_cache
            )
            predicted_options.append(predicted_option)
            match_methods.append(match_method)
            
        item["predicted_options"] = predicted_options
        item["match_methods"] = match_methods
        
        # 去除 -1, 即判断不出来的回答
        predicted_options = [option for option in predicted_options if option != -1]
        
        # 投票确定最终预测答案, 多个选项个数一样, 随机选一个
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
        
        item['final_predicted_option'] = final_predicted_option
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
    parser.add_argument('--config', default="config/nextqa_new_tool.yaml",type=str)
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    input_file_list = [
        # "archive/output/nextqa/results_20250420_204303.json",
        "output/nextqa/results_20250421_202125.json",
        # "output/nextqa/results_20250421_212143.json",
        "output/nextqa/results_20250420_222432.json",
        "output/nextqa/results_20250420_230747.json",
    ]
    
    output_file = "eval/nextqa/ensemble_1.json"

    with open(input_file_list[0], 'r') as f:
        data = json.load(f)
    data_len = len(data)
    for i in range(1, len(input_file_list)):
        extend_file = input_file_list[i]
        with open(extend_file, 'r') as f:
            extend_data = json.load(f)
        assert len(extend_data) == data_len
        for j in range(data_len):
            data[j]['answers'].extend(extend_data[j]['answers'])

    # LLM for rephrase
    eval_llm = ChatOpenAI(
        model=conf.openai.EVAL_MODEL_NAME,
        temperature=0.0,
        api_key=conf.openai.GPT_API_KEY,
        base_url=conf.openai.PROXY
    )
    # cache
    if conf.eval.use_cache and os.path.exists(conf.eval.eval_cache_file):
        print("loading cache...")
        with open(conf.eval.eval_cache_file, "rb") as f:
            llm_cache = pickle.load(f)
    else:
        llm_cache = {}

    main(data, output_file, conf, eval_llm, llm_cache)

    print(f"Output saved to {output_file}.")

