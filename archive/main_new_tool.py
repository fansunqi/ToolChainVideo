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
from util import save_to_json
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
)


mannual_cache = None
mannual_cache_file = None

current_video_dir = None
current_video_name = None


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
    def inference(self, input):
       
        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db") 
        question = input

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)

        print(
            f"\nProcessed TemporalTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


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
    def inference(self, input):

        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db")
        question = input

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)

        print(
            f"\nProcessed CountingTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


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
    def inference(self, input):
        
        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db")
        question = input

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        
        # TODO 了解一下下面这几行是如何运作的
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)
        
        print(
            f"\nProcessed ReasonFinder, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


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
    def inference(self, input):
        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db")
        question = input

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        
        # sample_rows_in_table_info 的主要作用是控制在生成数据库表的元信息（table_info）时，是否包含表中的示例数据行。
        # 如果设置了该参数，langchain 会从每个表中抽取指定数量的行，并将这些行作为表信息的一部分，提供给语言模型（LLM）进行推理。
        
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)

        print(
            f"\nProcessed HowSeeker, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


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
    def inference(self, input):
        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db")
        question = input

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        
        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)

        print(
            f"\nProcessed DescriptionTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


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
    def inference(self, input):
        self.sql_path = os.path.join(current_video_dir, current_video_name + ".db")
        question = input

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        
        try:
            result = db_chain.invoke(question)
        except:
            result ="There is an error. Try to ask the question in a different way."
        
        # result = db_chain.invoke(question)

        print(
            f"\nProcessed DefaultTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


############Memory Bulider#########
class VideoTemporalUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = TemporalBase(device=self.device, config=self.config)

    @prompts(
        name="VideoTemporalUnderstanding",
        description="useful when you need to process temporal information in videos."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is /data/videos/xxx.mp4#Are the men happy today?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result
        
        step = self.config.memory.step
        self.basemodel.run_on_video(video_path, step)
        result = "successfully built"
        print(
            f"\nProcessed VideoTemporalUnderstanding, Input Video: {video_path}, "
        )
        return result


class VideoInstanceUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = InstanceBase(device=self.device, config=self.config)

    @prompts(
        name="VideoInstanceUnderstanding",
        description="useful when you need to understand the instance information in videos."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is /data/videos/xxx.mp4#Are the men happy today?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result
        
        step = self.config.memory.step
        self.basemodel.run_on_video(video_path,question,step)
        result = "successfully built"
        print(
            f"\nProcessed VideoInstanceUnderstanding, Input Video: {video_path}, "
        )
        return result



class MemeryBuilder:
    def __init__(self, load_dict, config):
        print(f"Initializing MemoryBuilder, load_dict={load_dict}")

        self.config = config
        self.models = {}
        self.examplesel = CustomExampleSelector()

        # 根据 load_dict 动态选择, 加载模型
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device,config=self.config)

        # Load Template Foundation Models
        # TODO 没懂这段代码是什么意思
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k
                    for k in inspect.signature(module.__init__).parameters.keys()
                    if k != "self"
                }
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names}
                    )

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )

        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )

        # self.memory = ConversationBufferMemory(memory_key="chat_history")

    # db_agent 从 tool 中被分了出来
    def init_db_agent(self):
        tools = []
        self.db_model_list = []

        memory_load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.memory.memory_list}
        for class_name, device in memory_load_dict.items():
            self.db_model_list.append(globals()[class_name](device=device,config=self.config))
        
        for instance in self.db_model_list:
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )

        llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )

        memory = ConversationBufferMemory(memory_key="chat_history")
        self.db_agent = initialize_agent(
            tools,
            llm,
            agent="conversational-react-description",
            verbose=True,
            memory=memory,
        )

    def run_db_agent(self, video_path, question, with_two_mem):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + ".db")

        # 移除之前的数据库
        if os.path.exists(sql_path):  
            os.remove(sql_path)

        # TODO 这个分支有什么不同
        if with_two_mem:
            self.db_model_list[0].inference(video_path + "#"+ question)
            self.db_model_list[1].inference(video_path + "#"+ question)
        else:
            # 自己编造之前的对话
            Human_prompt = f"provide a video from {video_path}. You must use at least one tool to finish following tasks, rather than directly imagine from my description. If you understand, say 'Received'."
            AI_prompt = f"Received."

            # 存入对话 buffer
            self.db_agent.memory.save_context(
                {"input": Human_prompt}, {"output": AI_prompt}
            )

            self.db_agent.run(input=question.strip())

            video_dir = os.path.dirname(video_path)
            video_name = os.path.basename(video_path).split(".")[0]
            self.sql_path = os.path.join(video_dir, video_name + ".db")
            conn = sqlite3.connect(self.sql_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='instancedb';"
            )
            rows_1 = cursor.fetchall()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='temporaldb';"
            )
            rows_2 = cursor.fetchall()
            if len(rows_1) == 0 and len(rows_2) == 0:
                self.db_model_list[0].inference(video_path + "#"+ question)
                self.db_model_list[1].inference(video_path + "#"+ question)

    def run_example(self, example):
        input = example["Input"]
        output = example["Output"]
        Human_prompt = f"Here is an example. The question is: {input}; The chain is:{output}; If you understand, say 'Received'."
        AI_prompt = f"Received."

        self.agent.memory.save_context({"input": Human_prompt}, {"output": AI_prompt})

        print(f" Current Memory: {self.agent.memory.load_memory_variables({})}")



def use_tool_calling_agent(
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
            ("human", "{input}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    # TODO 详细查阅文档的其它功能
    agent = create_tool_calling_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    # TODO 更改 prompt
    # TODO 1 加上 tool descriptions, 能不能用到内部 format
    query = f"""
    Regarding a given video from {video_filename}, use tools to answer the following questions as best you can.
    Question: {input_question}
    """
    
    step_idx = 0
    output = None
    
    if use_cache and (query in mannual_cache):
        # 缓存命中
        print("\nCache hit!")
        steps = mannual_cache[query]
        output = steps[-1].get('output')
    else:
        # 缓存未命中
        print("\nCache miss. Calling API...")
        steps = []
        for step in agent_executor.stream({"input": query}):
            step_idx += 1
            # print(f"\nagent_iterator step: {step_idx}")
            # print(step)
            steps.append(step)
            output = step.get('output')
            # pdb.set_trace()
        
        mannual_cache[query] = steps
        # 保存缓存
        print("\nSaving cache...")
        with open(mannual_cache_file, "wb") as f:
            pickle.dump(mannual_cache, f)
        
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa_new_tool.yaml",type=str)                           
    opt = parser.parse_args()

    vq_conf = OmegaConf.load(opt.config)
    conf = OmegaConf.load(vq_conf.inference_config_path)

    seed_everything(vq_conf.seed) 
    
    # mannual_cache
    print("\nloading mannual_cache...")
    mannual_cache_file = vq_conf.mannual_cache_file
    if os.path.exists(mannual_cache_file):
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        mannual_cache = {}
    

    load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.tool.tool_list}
    # {'TemporalTool': 'cpu', 'CountingTool': 'cpu', 'ReasonFinder': 'cpu', 'HowSeeker': 'cpu', 'DescriptionTool': 'cpu', 'DefaultTool': 'cpu'}

    bot = MemeryBuilder(load_dict=load_dict, config=conf)

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
        
        print(f"\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        question = data["question"].capitalize()  # 首字母大写
        options = [data['optionA'], data['optionB'], data['optionC'], data['optionD'], data['optionE']]
        question_w_options = f"{question}? Choose your answer from below selections: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."

        if not vq_conf.skip_mem_build:  
            # if you have built the memory, you can skip this step by setting skip_mem_build=True
            bot.init_db_agent()
            bot.run_db_agent(video_path, question_w_options, vq_conf.with_two_mem)
        
        current_video_dir = os.path.dirname(video_path)
        current_video_name = os.path.basename(video_path).split(".")[0]
        
        # try:
        answers = {}
        answers["good_anwsers"] = []
        answers["bas_anwsers"] = []
        answer = use_tool_calling_agent(video_filename=video_path,
                                        input_question=question_w_options,
                                        llm=llm,
                                        tools=bot.tools,
                                        use_cache=vq_conf.use_cache)
        answers["good_anwsers"].append(answer)
        # except Exception as e:
        #     print(f"Error:{e}")
        #     answers = "Error"

        result_dict = data
        result_dict["question_w_options"] = question_w_options
        result_dict["answers"] = answers
        all_results.append(result_dict)

    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{vq_conf.output_file[:-5]}_{timestamp}.json"

    save_to_json(all_results, output_file)
    print(f"{str(len(all_results))}results saved")



# TODO: log，哪些东西可以放到 log 里面
# TODO: 深入看一下 memory, 优化 memory
# TODO: PromptTemplate 和 ChatPromptTemplate 的内部格式化
    
    
    
