# Licensed under the BSD 3-Clause License.
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(f"./")
sys.path.append(f"../")

import random, inspect
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
import sqlite3
import pickle
import datetime
from tqdm import tqdm
from dataset import get_dataset
from util import save_to_json, adjust_video_resolution
import pdb

import langchain
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")
set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent

### For langgraph iteration
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate


from project.TemporalUnderstanding import TemporalBase
from project.InstanceUnderstanding import InstanceBase
from project.ExampleSelector import CustomExampleSelector


from project.sql_template import (
    _sqlite_prompt,
    COUNTING_EXAMPLE_PROMPT,
    PROMPT_SUFFIX,
    TEMPORAL_EXAMPLE_PROMPT,
    REASONFINDER_ADDITION_PROMPT,
    HOWSEEKER_ADDITION_PROMPT,
    DESCRIP_EXAMPLE_PROMPT,
    DESCRIP_ADDITION_PROMPT,
    TOOLS_RULE,
    QUERY_PREFIX
)


mannual_cache = None
mannual_cache_file = None

current_video_dir = None
current_video_name = None

# 获取当前时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


TOOL_INPUT_ERROR = "There is an error in the input of the tool, please check the input and try again."
TOOL_PROCESS_ERROR = "There is an error. Try to ask the question in a different way."


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

    
class TemporalTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + TEMPORAL_EXAMPLE_PROMPT + PROMPT_SUFFIX,     # 各种 tool 的不同就在于这里的 prompt 模块不同
        )

    @prompts(
        name = "TemporalTool",
        description = "Useful when you need to process temporal information in videos."
        "The input to this tool must be a question. For example, the input is 'what is he talking about when a girl is playing violin?'",
    )
    def inference(self, question):
        print("\nCall TemporalTool")

        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        
        # TODO 进一步了解 prompt 格式化的过程      
        # sample_rows_in_table_info 的主要作用是控制在生成数据库表的元信息（table_info）时，是否包含表中的示例数据行。
        # 如果设置了该参数，langchain 会从每个表中抽取指定数量的行，并将这些行作为表信息的一部分，提供给语言模型（LLM）进行推理。
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        
        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR

        print("\nProcessed TemporalTool.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        # TODO assert 检查返回的是字符串
        # TODO 查看返回下面哪个比较好？
        return result
        # return db_chain_output


class CountingTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + COUNTING_EXAMPLE_PROMPT+ PROMPT_SUFFIX,
        )

    @prompts(
        name = "CountingTool",
        description = "Useful when you need to count object number."
        "The input to this tool must be a question. For example, the input is 'How many fish are here?'",
    )
    def inference(self, question):
        print("\nCall CountingTool")
        
        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR

        print("\nProcessed CountingTool.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        return result
        # return db_chain_output


class ReasonFinder:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + REASONFINDER_ADDITION_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name="ReasonFinder",
        description="Useful when you need to find reasons or explanations."
        "The input to this tool must be a question. For example, the input is 'Why she is crying?'",
    )
    def inference(self, question):
        print("\nCall ReasonFinder")

        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR
        
        print("\nProcessed ReasonFinder.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        return result
        # return db_chain_output


class HowSeeker:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template=_sqlite_prompt + HOWSEEKER_ADDITION_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name = "HowSeeker",
        description = "useful when you need to find methods or steps to accomplish a task."
        "The input to this tool must be a question. For example, the input is 'How did the children eat food?'",
    )
    def inference(self, question):
        print("\nCall HowSeeker")

        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )   
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR

        print("\nProcessed HowSeeker.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        return result
        # return db_chain_output


class DescriptionTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template=_sqlite_prompt + DESCRIP_ADDITION_PROMPT + DESCRIP_EXAMPLE_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name = "DescriptionTool",
        description = "Useful when you need to describe the content of a video, e.g. the audio in the video, the subtitles, the on-screen content, etc."
        "The input to this tool must be a question. For example, the input is 'What's in the video?'",
    )
    def inference(self, question):
        print("\nCall DescriptionTool")
        
        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        
        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR

        print("\nProcessed DescriptionTool.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        return result
        # return db_chain_output

# TODO 可以把 DefaultTool 换成 SQLDatabase toolkit
class DefaultTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + PROMPT_SUFFIX,
        )

    @prompts(
        name = "DefaultTool",
        description = "Useful when other tools can't solve the problem corresponding to the video."
        "The input to this tool must be a question. For example, the input is 'Are the men happy today?'",
    )
    def inference(self, question):
        print("\nCall DefaultTool")

        db_version = self.config.memory.db_version
        self.sql_path = os.path.join(current_video_dir, current_video_name + "_" + str(db_version) + ".db")
        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        
        # try:
        db_chain_output = db_chain.invoke(question)
        result = db_chain_output['result']
        input_question = db_chain_output['query']
        # except:
        #     print(TOOL_PROCESS_ERROR)
        #     return TOOL_PROCESS_ERROR

        print("\nProcessed DefaultTool.")
        print(f"Input Video: {video_path}")
        print(f"Original Question: {question}")
        print(f"Input Question: {input_question}")
        print(f"Output Answer: {result}")
        
        return result
        # return db_chain_output


class VideoTemporalUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = TemporalBase(device=self.device, config=self.config)

    @prompts(
        name="VideoTemporalUnderstanding",
        description="Useful when you need to process temporal information in videos. Must be called before access to temporaldb database."
    )
    def inference(self, input):
        print("\nCall VideoTemporalUnderstanding")
        
        step = self.config.memory.step
        db_version = self.config.memory.db_version
        video_path = os.path.join(current_video_dir, current_video_name + ".mp4")
        
        self.basemodel.run_on_video(video_path, step, db_version)
        
        result = "Successfully built temporaldb database."
        print(f"\nProcessed VideoTemporalUnderstanding, video_path = {video_path}")
        return result


class VideoInstanceUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = InstanceBase(device=self.device, config=self.config)

    @prompts(
        name="VideoInstanceUnderstanding",
        description="Useful when you need to understand the instance information in videos. Must be called before access to instancedb database."
    )
    def inference(self, input):
        print("\nCall VideoInstanceUnderstanding")
        
        step = self.config.memory.step
        db_version = self.config.memory.db_version
        video_path = os.path.join(current_video_dir, current_video_name + ".mp4")
        
        self.basemodel.run_on_video(video_path, None, step, db_version)
        result = "Successfully built instancedb database."
        print(f"\nProcessed VideoInstanceUnderstanding, video_path = {video_path}")
        return result
    
    
def ToolChainReasoning(
    video_filename, 
    input_question, 
    llm, 
    tools, 
    use_cache=True,
):
    # TODO 考虑之前选择工具的历史
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("placeholder", "{messages}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    def _modify_state_messages(state: AgentState):
        return prompt.invoke({"messages": state["messages"]}).to_messages()
    
    app = create_react_agent(llm, tools, state_modifier=_modify_state_messages)
    
    # TODO 更改 prompt
    query = QUERY_PREFIX + input_question + '\n' + TOOLS_RULE
    
    
    if use_cache and (query in mannual_cache):
        print("\nCache hit!")
        steps = mannual_cache[query]
    else:
        print("\nCache miss. Calling API...")
        steps = []
        step_idx = 0
        for step in app.stream({"messages": [("human", query)]}, stream_mode="updates"):
            
            # pdb.set_trace()
            
            step_idx += 1
            steps.append(step)
                
        mannual_cache[query] = steps
        print("\nSaving cache...")
        with open(mannual_cache_file, "wb") as f:
            pickle.dump(mannual_cache, f)
        
    try:
        output = steps[-1]['agent']['messages'][0].content
    except:
        output = None
    
    print(f"\nToolChainOutput: {output}") 
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa_run_build.yaml",type=str)                           
    opt = parser.parse_args()

    vq_conf = OmegaConf.load(opt.config)
    conf = OmegaConf.load(vq_conf.inference_config_path)

    seed_everything(vq_conf.seed) 
    
    # mannual LLM cache
    mannual_cache_file = vq_conf.mannual_cache_file
    if os.path.exists(mannual_cache_file):
        print(f"\nLoading LLM cache from {mannual_cache_file}...")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"\nCreating LLM cache: {mannual_cache_file}...")
        mannual_cache = {}
    
    
    sql_tool_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.tool.tool_list}
    mem_tool_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.memory.memory_list}
    
    # 根据 load_dict 动态选择, 加载模型
    tool_instance = {}
    for class_name, device in sql_tool_dict.items():
        tool_instance[class_name] = globals()[class_name](device=device, config=conf)
    for class_name, device in mem_tool_dict.items():
        tool_instance[class_name] = globals()[class_name](device=device, config=conf)

    # 把 tool_instance 的 inference 方法加到 tools 中去
    tools = []
    for instance in tool_instance.values():
        for e in dir(instance):
            if e.startswith("inference"):
                func = getattr(instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))

    # 数据集
    quids_to_exclude = vq_conf["quids_to_exclude"] if "quids_to_exclude" in vq_conf else None
    num_examples_to_run = vq_conf["num_examples_to_run"] if "num_examples_to_run" in vq_conf else -1
    start_num = vq_conf["start_num"] if "start_num" in vq_conf else 0
    specific_quids = vq_conf["specific_quids"] if "specific_quids" in vq_conf else None
    dataset = get_dataset(vq_conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)
    all_results = []
    
    # 用于 planning tools 的 llm
    llm = ChatOpenAI(
        api_key = conf.openai.GPT_API_KEY,
        model = conf.openai.GPT_MODEL_NAME,
        temperature = 0,
        base_url = conf.openai.PROXY
    )

    for data in tqdm(dataset):
        
        print(f"\n\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        question = data["question"].capitalize()  # 首字母大写
        options = [data['optionA'], data['optionB'], data['optionC'], data['optionD'], data['optionE']]
        question_w_options = f"{question}? Choose your answer from below selections: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."
        
        current_video_dir = os.path.dirname(video_path)
        current_video_name = os.path.basename(video_path).split(".")[0]
        track_res_base = "intermediate_res/track_res"
        track_res_path = os.path.join(track_res_base, current_video_name, current_video_name + ".avi")
 
        #### trim
        adjust_video_resolution(video_path)
        
        #### ToolChainReasoning   
        # try:
        answers = {}
        answers["good_anwsers"] = []
        answers["bas_anwsers"] = []
        answer = ToolChainReasoning(video_filename=video_path,
                                    input_question=question_w_options,
                                    llm=llm,
                                    tools=tools,
                                    use_cache=vq_conf.use_cache)
        answers["good_anwsers"].append(answer)
        # except Exception as e:
        #     print(f"\nError:{e}")
        #     answers = "Error"

        result_dict = data
        result_dict["question_w_options"] = question_w_options
        result_dict["answers"] = answers
        all_results.append(result_dict)

    
    output_file = f"{vq_conf.output_file[:-5]}_{timestamp}.json"
    save_to_json(all_results, output_file)
    print(f"\n{str(len(all_results))} results saved")



# TODO: log，哪些东西可以放到 log 里面
# TODO: 深入看一下 memory, 优化 memory
# TODO: PromptTemplate 和 ChatPromptTemplate 的内部格式化

# TODO 看一下 coco.txt 到底是怎么样的
# TODO: 修复下面这个 error:
'''
Traceback (most recent call last):
  File "/home/fsq/video_agent/ToolChainVideo/./scripts/main.py", line 792, in <module>
    bot.run_db_agent(video_path, question_w_options, vq_conf.with_two_mem)
  File "/home/fsq/video_agent/ToolChainVideo/./scripts/main.py", line 627, in run_db_agent
    self.db_model_list[0].inference(video_path + "#"+ question)
  File "/home/fsq/video_agent/ToolChainVideo/./scripts/main.py", line 532, in inference
    self.basemodel.run_on_video(video_path,question,step)
  File "/home/fsq/video_agent/ToolChainVideo/./project/InstanceUnderstanding.py", line 341, in run_on_video
    self.inital_database()
  File "/home/fsq/video_agent/ToolChainVideo/./project/InstanceUnderstanding.py", line 108, in inital_database
    result_list = []
ValueError: invalid literal for int() with base 10: '0.0583333'
'''
    
    
    
