import os
import sys
import pdb
import datetime


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


from dataset import get_dataset

from util import (
    save_to_json, 
    adjust_video_resolution,
    backup_file,
    load_cache, 
)


from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

from tools.yolo_tracker import YOLOTracker
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector
from tools.image_qa import ImageQA
from tools.temporal_grounding import TemporalGrounding
from tools.image_grid_qa import ImageGridQA
from tools.summarizer import Summarizer

from visible_frames import get_video_info, VisibleFrames

from reasoning import (
    langgraph_reasoning,
    spatiotemporal_reasoning,
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa_st.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    backup_file(opt, conf, timestamp)

    if conf.to_txt:
        log_path = os.path.join(conf.output_path, f"log_{timestamp}.txt")
        f = open(log_path, "w")
        sys.stdout = f
    
    mannual_cache_file = conf.mannual_cache_file
    mannual_cache = load_cache(mannual_cache_file)

    tool_instances, tools = get_tools(conf)
    
    tool_planner_llm = ChatOpenAI(
        api_key = conf.openai.GPT_API_KEY,
        model = conf.openai.GPT_MODEL_NAME,
        temperature = 0,
        base_url = conf.openai.PROXY
    )

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

        # trim
        adjust_video_resolution(video_path)

        video_info = get_video_info(video_path)
        init_video_stride = int(video_info["fps"] * conf.init_interval_sec)

        print(question_w_options)
        
        for try_count in range(try_num):

            # visible_frames = VisibleFrames(video_path=video_path, init_video_stride=init_video_stride)
            visible_frames = VisibleFrames(video_path=video_path, init_video_stride=None)
            
            for tool_instance in tool_instances:
                tool_instance.set_frames(visible_frames)
                tool_instance.set_video_path(video_path)

            # TODO 各种工具也可以加上一个 set_question 功能, 使得 question 不用再占据 input 的位置
            # TODO 简化，统一下面的代码
            
            if conf.try_except_mode:
                try:
                    if conf.reasoning_mode == "langgrah":
                        tool_chain_output = langgraph_reasoning(
                            input_question=question_w_options,
                            llm=tool_planner_llm,
                            tools=tools,
                            recursion_limit=conf.recursion_limit,
                            use_cache=conf.use_cache,
                            mannual_cache=mannual_cache,
                            mannual_cache_file=mannual_cache_file
                        )
                    elif conf.reasoning_mode == "st":
                        tool_chain_output = spatiotemporal_reasoning(
                            question=question,
                            question_w_options=question_w_options,
                            llm=tool_planner_llm,
                            tools=tool_instances,
                            recursion_limit=conf.recursion_limit,
                            use_cache=conf.use_cache,
                            mannual_cache=mannual_cache,
                            mannual_cache_file=mannual_cache_file
                        )
                    else:
                        raise KeyError("conf.reasoning_mode error")
                except Exception as e:
                    print(f"Error: {e}")
                    tool_chain_output = "Error"
            else:
                if conf.reasoning_mode == "langgrah":
                    tool_chain_output = langgraph_reasoning(
                        input_question=question_w_options,
                        llm=tool_planner_llm,
                        tools=tools,
                        recursion_limit=conf.recursion_limit,
                        use_cache=conf.use_cache,
                        mannual_cache=mannual_cache,
                        mannual_cache_file=mannual_cache_file
                    )
                elif conf.reasoning_mode == "st":
                     tool_chain_output = spatiotemporal_reasoning(
                        question=question,
                        question_w_options=question_w_options,
                        llm=tool_planner_llm,
                        tools=tool_instances,
                        recursion_limit=conf.recursion_limit,
                        use_cache=conf.use_cache,
                        mannual_cache=mannual_cache,
                        mannual_cache_file=mannual_cache_file
                    )
                else:
                    raise KeyError("conf.reasoning_mode error")

            result["answers"].append(tool_chain_output)

        all_results.append(result)

    output_file = os.path.join(conf.output_path, f"results_{timestamp}.json")
    save_to_json(all_results, output_file)
    print(f"\n{str(len(all_results))} results saved")   

    if conf.to_txt:
        sys.stdout = sys.__stdout__
        f.close()

    