import os
import pdb
import math
import numpy as np

import langchain
from langchain_community.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from project.ExampleSelector import CustomExampleSelector
from project.PromptTemplate import general_template

from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt


class ReThinking(object):
    def __init__(
        self,
        llm,
        tools,
        good_base_reward=1.0,
        bad_base_reward=-1.0,
        decay_rate=0.5,
        template=general_template,
        use_example=False,
    ):
        self.llm = llm
        self.tools = tools
        self.template = template
        self.use_example = use_example
        self.video_name = ""
        self.question = ""
        self.examplesel = CustomExampleSelector() if self.use_example else None
        self.examples = ""
        self.total_step = 0
        self.parsing_error_alarm = "Failed to parse"
        self.good_base_reward = good_base_reward
        self.bad_base_reward = bad_base_reward
        self.decay_rate = decay_rate
        self.possible_anwsers = []

    def run(
        self,
        video_name,
        question,
        possible_anwsers=[],
        max_answer=3,
        max_try=8,
        use_example=False,
        quid=None,
    ):
        
        self.use_example = use_example
        anwsers = {"good_anwsers": [], "bad_anwsers": []}
        num_try = 0
        
        # 进入主循环的迭代
        while (
            len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) < max_answer
            and num_try < max_try
        ):

            print(f"\n\nTree {num_try}:\n\n") 
    

            num_try += 1
            num_anwser = len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) + 1
            
            print(f"\n\nAnswer {num_anwser} - Try {num_try}:\n\n")
            

            is_good_result = True
            
            final_answer = self.use_tool_calling_agent(video_filename=video_name,
                                        input_question=question)
            
            
            if is_good_result:
                # 只有这里才会输出答案
                anwsers["good_anwsers"].append(final_answer)
            else:
                anwsers["bad_anwsers"].append(final_answer)

        return anwsers
    
    def use_tool_calling_agent(self, video_filename, input_question, ancestor_history=None, children_history=None):
        # 先不考虑 ancestor_history 和 children_history 这两项
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        # TODO 首先是看 agent 能否看到 tool 的描述
        agent = create_tool_calling_agent(self.llm, self.tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools)
        
        query = f"""
        Regarding a given video from {video_filename}, use tools to answer the following questions as best you can.
        Question: {input_question}
        """
        
        step_idx = 0
        output = None
        for step in agent_executor.stream({"input": query}):
            step_idx += 1
            print(f"\nagent_iterator step: {step_idx}")
            print(step)
            output = step.get('output')
        
        return output