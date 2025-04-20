import os
import sys
import pdb
import datetime
import shutil
import pickle
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

seed = 12345
import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
    ASSISTANT_ROLE,
)

from dataset import get_dataset

from util import save_to_json, adjust_video_resolution

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

from tools.yolo_tracker import YOLOTracker
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector

from visible_frames import get_video_info, VisibleFrames


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TO_TXT = True
TRY_EXCEPT_MODE = True


def backup_file(opt, conf):
    # 将 main.py 文件自身和 opt.config 文件复制一份存储至 conf.output_path
    current_script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    shutil.copy(current_script_path, os.path.join(conf.output_path, f"main_{timestamp}.py"))
    # 复制 opt.config 文件到输出目录
    config_basename = os.path.basename(opt.config).split('.')[0]
    shutil.copy(opt.config, os.path.join(conf.output_path, f"{config_basename}_{timestamp}.yaml"))   
    

def load_cache(mannual_cache_file):
    if os.path.exists(mannual_cache_file):
        print(f"Loading LLM cache from {mannual_cache_file}...\n")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"Creating LLM cache: {mannual_cache_file}...\n")
        mannual_cache = {}
    return mannual_cache


def save_cache(mannual_cache, query, steps, mannual_cache_file):
    mannual_cache[query] = steps
    print("\nSaving cache...")
    with open(mannual_cache_file, "wb") as f:
        pickle.dump(mannual_cache, f)


def get_tools(conf):
    tool_list = conf.tool.tool_list
    tool_instances = []
    for tool_name in tool_list:
        tool_instances.append(globals()[tool_name](conf))
    print(f"tool_instances: {str(tool_instances)}")
    
    tools = []
    for tool_instance in tool_instances:
        for e in dir(tool_instance):
            if e.startswith("inference"):
                func = getattr(tool_instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))
    print(f"tools: {str(tools)}")
    
    return tool_instances, tools


def tool_chain_reasoning( 
    input_question, 
    llm, 
    tools,
    recursion_limit=24,  
    use_cache=True,
    mannual_cache=None,
    mannual_cache_file=None
):
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful assistant."),
    #         ("placeholder", "{messages}"),
    #         ("placeholder", "{agent_scratchpad}"),
    #     ]
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSISTANT_ROLE),
            ("placeholder", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    def _modify_state_messages(state: AgentState):
        return prompt.invoke({"messages": state["messages"]}).to_messages()
    
    tool_planner = create_react_agent(llm, tools, state_modifier=_modify_state_messages)
    
    query = QUERY_PREFIX + input_question + '\n\n' + TOOLS_RULE

    # query = QUERY_PREFIX + input_question
    
    if use_cache and (query in mannual_cache):
        print("\nCache hit!")
        steps = mannual_cache[query]
    else:
        print("\nCache miss. Calling API...")
        steps = []
    
        for step in tool_planner.stream(
            {"messages": [("human", query)]}, 
            {"recursion_limit": recursion_limit},
                stream_mode="values"):

            step_message = step["messages"][-1]

            if isinstance(step_message, tuple):
                print(step_message)
            else:
                step_message.pretty_print()
        
            steps.append(step)
        
        save_cache(mannual_cache, query, steps, mannual_cache_file)    
 
    try:
        output = steps[-1]["messages"][-1].content
    except:
        output = None
    
    print(f"\nToolChainOutput: {output}") 
    return output




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa_new_tool.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    backup_file(opt, conf)

    if TO_TXT:
        # 指定输出 log
        log_path = os.path.join(conf.output_path, f"log_{timestamp}.txt")
        f = open(log_path, "w")
        # 重定向标准输出
        sys.stdout = f
    
    # mannual LLM cache
    mannual_cache_file = conf.mannual_cache_file
    mannual_cache = load_cache(mannual_cache_file)

    tool_instances, tools = get_tools(conf)
    
    tool_planner_llm = ChatOpenAI(
        api_key = conf.openai.GPT_API_KEY,
        model = conf.openai.GPT_MODEL_NAME,
        temperature = 0,
        base_url = conf.openai.PROXY
    )

    # 数据集
    quids_to_exclude = conf["quids_to_exclude"] if "quids_to_exclude" in conf else None
    num_examples_to_run = conf["num_examples_to_run"] if "num_examples_to_run" in conf else -1
    start_num = conf["start_num"] if "start_num" in conf else 0
    specific_quids = conf["specific_quids"] if "specific_quids" in conf else None
    dataset = get_dataset(conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)

    try_num = conf.try_num
    all_results = []

    for data in tqdm(dataset):

        print(f"\n\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        question = data["question"].capitalize()  # 首字母大写
        options = [data['optionA'], data['optionB'], data['optionC'], data['optionD'], data['optionE']]
        question_w_options = f"{question}? Choose your answer from below options: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."

        result = data
        result["answers"] = []
        result["question_w_options"] = question_w_options

        #### trim
        adjust_video_resolution(video_path)

        # 建立 VisibleFrames
        video_info = get_video_info(video_path)
        init_video_stride = int(video_info["fps"] * conf.init_interval_sec)
        visible_frames = VisibleFrames(video_path=video_path, init_video_stride=init_video_stride)


        for try_count in range(try_num):
            
            for tool_instance in tool_instances:
                tool_instance.set_frames(visible_frames)

            # TODO 各种工具也可以加上一个 set_question 功能
            
            if TRY_EXCEPT_MODE:
                try:
                    tool_chain_output = tool_chain_reasoning(
                        input_question=question_w_options,
                        llm=tool_planner_llm,
                        tools=tools,
                        recursion_limit=conf.recursion_limit,
                        use_cache=conf.use_cache,
                        mannual_cache=mannual_cache,
                        mannual_cache_file=mannual_cache_file
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    tool_chain_output = "Error"
            else:
                tool_chain_output = tool_chain_reasoning(
                    input_question=question_w_options,
                    llm=tool_planner_llm,
                    tools=tools,
                    recursion_limit=conf.recursion_limit,
                    use_cache=conf.use_cache,
                    mannual_cache=mannual_cache,
                    mannual_cache_file=mannual_cache_file
                )

            result["answers"].append(tool_chain_output)

            # TODO 检查 tool_chain_output 是否真正选出来了答案

        all_results.append(result)

    output_file = os.path.join(conf.output_path, f"results_{timestamp}.json")
    save_to_json(all_results, output_file)
    print(f"\n{str(len(all_results))} results saved")   

    if TO_TXT:
            # 恢复标准输出
        sys.stdout = sys.__stdout__
        f.close()



# TODO 可视化 langgraph 工具图


    