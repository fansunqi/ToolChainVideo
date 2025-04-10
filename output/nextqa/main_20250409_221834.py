import os
import sys
import datetime
import shutil
import pickle
import argparse
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
)

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
TO_TXT = False


def backup_file(opt, conf):
    # 将 main.py 文件自身和 opt.config 文件复制一份存储至 conf.output_path
    current_script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    shutil.copy(current_script_path, os.path.join(conf.output_path, f"main_{timestamp}.py"))
    # 复制 opt.config 文件到输出目录
    config_basename = os.path.basename(opt.config).split('.')[0]
    shutil.copy(opt.config, os.path.join(conf.output_path, f"{config_basename}_{timestamp}.yaml"))   
    

def load_cache(conf):
    mannual_cache_file = conf.mannual_cache_file
    if os.path.exists(mannual_cache_file):
        print(f"\nLoading LLM cache from {mannual_cache_file}...")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"\nCreating LLM cache: {mannual_cache_file}...")
        mannual_cache = {}
    return mannual_cache


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
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("placeholder", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    def _modify_state_messages(state: AgentState):
        return prompt.invoke({"messages": state["messages"]}).to_messages()
    
    tool_planner = create_react_agent(llm, tools, state_modifier=_modify_state_messages)
    
    query = QUERY_PREFIX + input_question + '\n\n' + TOOLS_RULE
    
    steps = []
    step_idx = 0
    
    # TODO 研究一下这里的 stream_mode
    for step in tool_planner.stream(
        {"messages": [("human", query)]}, 
        {"recursion_limit": recursion_limit},
            stream_mode="values"):

        step_message = step["messages"][-1]

        if isinstance(step_message, tuple):
            print(step_message)
        else:
            step_message.pretty_print()
        
        # 是不是需要在每一步中手动调用 frame_selector
        
        step_idx += 1
        steps.append(step)
 
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
    mannual_cache = load_cache(conf)

    tool_instances, tools = get_tools(conf)
    
    # 视频路径
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    question_w_options = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
    
    tool_planner_llm = ChatOpenAI(
        api_key = conf.openai.GPT_API_KEY,
        model = conf.openai.GPT_MODEL_NAME,
        temperature = 0,
        base_url = conf.openai.PROXY
    )

    # 建立 VisibleFrames
    video_info = get_video_info(video_path)
    init_video_stride = int(video_info["fps"] * conf.init_interval_sec)
    visible_frames = VisibleFrames(video_path=video_path, init_video_stride=init_video_stride)
    
    for tool_instance in tool_instances:
        tool_instance.set_frames(visible_frames)
    
    tool_chain_output = tool_chain_reasoning(
        input_question=question_w_options,
        llm=tool_planner_llm,
        tools=tools,
        recursion_limit=24,
    )

    print(tool_chain_output)

    if TO_TXT:
            # 恢复标准输出
        sys.stdout = sys.__stdout__
        f.close()



    