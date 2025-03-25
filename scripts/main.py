# Licensed under the MIT License.
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(f"./")
sys.path.append(f"../")

import random, math, cv2, inspect, tempfile, csv
import torch
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import argparse
from omegaconf import OmegaConf
import sqlite3

import langchain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain_core.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.cache import SQLiteCache

# 启用缓存 (SQLite 方式), 可以去掉这行代码进行对比
langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")

from project.TemporalUnderstanding import TemporalBase
from project.InstanceUnderstanding import InstanceBase
from project.ExampleSelector import CustomExampleSelector
from project.TreeSearch import ReThinking
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
from project.E2FGVI.Inpainter import Inpainter

import datetime
from tqdm import tqdm
from dataset import get_dataset
from util import save_to_json

import pdb


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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#What is he talking about when a girl is playing violin? ",
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
            result = db_chain.run(question)   # 自然语言自动化查询数据库
        except:
            result ="There is an error. Try to ask the question in a different way."

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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#How many fish are here? ",
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
            result = db_chain.run(question)
        
        # 报错返回
        except:
            result ="There is an error. Try to ask the question in a different way."

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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Why she is crying? ",
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
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#How did the children eat food? ",
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
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#What's in the video?",
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
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

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
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
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
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed DefaultTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result

class InpaintingTool:
    def __init__(self, device, config):
        self.device = device
        self.inpainter = Inpainter(device=device)

    @prompts(
        name = "InpaintingTool",
        description = "Useful when user want to inpaint something in the video."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Inpaint the men on the right.",
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

        video_name = os.path.basename(video_path).split(".")[0]
        imgseq_path = f"./demo/rvos_res/images/{video_name}/"
        maskseq_path = f"./demo/rvos_res/results/{video_name}/"

        # try:
        res_path = self.inpainter.main_worker(video_path = imgseq_path, mask_path = maskseq_path)
        result = f"Finish inpainting. The results are saved in {res_path}."
        # except:
        #     result = "There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed InpaintingTool, Input Video: {video_path}, Input Question: {question}, "
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
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
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
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
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

        self.memory = ConversationBufferMemory(memory_key="chat_history")

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

    def run_db_agent(self, video_path, question,with_two_mem):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + ".db")  # sqlite 数据库的路径


        if os.path.exists(sql_path):  # 如果数据库已经预先建好,就移除数据库
            os.remove(sql_path)

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



def run_a_video(
    MemoryBuilder,
    Planner,
    video_name,
    question,
    possible_anwsers=[],
    skip_mem_build=True,
    with_two_mem = True,
    use_example=False,
    max_answer=1,
    max_try=7,
    quid=None,
):
    if (
        not skip_mem_build
    ):  # if you have built the memory, you can skip this step by setting build_mem=False
        MemoryBuilder.init_db_agent()
        MemoryBuilder.run_db_agent(video_name, question, with_two_mem)

    anwsers = Planner.run(
        video_name,
        question,
        possible_anwsers=possible_anwsers,
        max_answer=max_answer,
        max_try=max_try,
        use_example=use_example,
        quid=quid,
    )
    print("Input video: ", video_name)
    print("Input question: ", question)
    print("The anwsers are:", anwsers)
    print("Total action steps: ", Planner.total_step)
    return anwsers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa.yaml",type=str)                           
    opt = parser.parse_args()

    vq_conf = OmegaConf.load(opt.config)
    conf = OmegaConf.load(vq_conf.inference_config_path)

    seed_everything(vq_conf.seed) 

    load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.tool.tool_list}
    # {'TemporalTool': 'cpu', 'CountingTool': 'cpu', 'ReasonFinder': 'cpu', 'HowSeeker': 'cpu', 'DescriptionTool': 'cpu', 'DefaultTool': 'cpu', 'InpaintingTool': 'cuda:0'}

    bot = MemeryBuilder(load_dict=load_dict, config=conf)

    planner = ReThinking(
        bot.llm, 
        bot.tools, 
        good_base_reward = conf.mcts_planner.good_base_reward, 
        bad_base_reward = conf.mcts_planner.bad_base_reward, 
        decay_rate = conf.mcts_planner.decay_rate,
    )

    quids_to_exclude = vq_conf["quids_to_exclude"] if "quids_to_exclude" in vq_conf else None
    num_examples_to_run = vq_conf["num_examples_to_run"] if "num_examples_to_run" in vq_conf else -1
    start_num = vq_conf["start_num"] if "start_num" in vq_conf else 0
    specific_quids = vq_conf["specific_quids"] if "specific_quids" in vq_conf else None
    dataset = get_dataset(vq_conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)
    all_results = []

    for data in tqdm(dataset):

        video_path = data["video_path"]
        question = data["question"].capitalize()  # 首字母大写
        options = [data['optionA'], data['optionB'], data['optionC'], data['optionD'], data['optionE']]
        formatted_question = f"{question}? Choose your answer from below selections: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."

        try:
            answers = run_a_video(
                bot,
                planner,
                video_path,
                formatted_question,
                skip_mem_build = vq_conf.skip_mem_build,
                with_two_mem = vq_conf.with_two_mem,
                max_try = vq_conf.max_try,
                max_answer = vq_conf.max_answer,
                quid = data["quid"],
            ) 

            # TODO
            # 如何去解析这个 answer ?
            # Input question:  How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five.
            # The anwsers are: {'good_anwsers': ['There are 29 children in the video.', 'D. two', '29 children'], 'bad_anwsers': []}
        except Exception as e:
            print(f"Error:{e}")
            print(data["quid"])
            # answers = "Error"
            sys.exit(1)  # 终止程序并返回状态码 1

        result_dict = data
        result_dict["formatted_question"] = formatted_question
        result_dict["answers"] = answers
        all_results.append(result_dict)

    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{vq_conf.output_file[:-5]}_{timestamp}.json"

    save_to_json(all_results, output_file)
    print(f"{str(len(all_results))}results saved")


# TODO: log，哪些东西可以放到 log 里面
# TODO: LLM cache
    
    
    
