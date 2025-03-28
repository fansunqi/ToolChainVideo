import os
import pdb
import math
import numpy as np

from langchain.agents.initialize import initialize_agent

from project.ExampleSelector import CustomExampleSelector
from project.PromptTemplate import general_template

from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

# 工具节点类
class Node(object):
    def __init__(self, value, father=None):
        self.father = father
        self.value = value        # 这个 value 值是 Node 的本体
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_ancestors(self, remove_root=True, child_first=False):
        ancestors = []
        current = self
        while current.father is not None:
            ancestors.append(current.father)
            current = current.father
        if len(ancestors) == 0:
            return []
        if remove_root:
            ancestors.pop()
        if not child_first:
            ancestors.reverse()
        return ancestors

    def get_descendants(self, remove_leaf=True):
        descendants = []
        if len(self.children) == 0:
            return []
        for child in self.children:
            if remove_leaf and child.is_leaf():
                continue
            descendants.append(child)
            descendants.extend(child.get_descendants(remove_leaf=remove_leaf))
        return descendants

# 工具树类
class MCSearchTree(object):
    def __init__(self, value):
        self.root = Node(value)
        self.current = self.root

    def traverse(self):
        self._traverse_helper(self.root)

    def _traverse_helper(self, node):
        print(node.value)
        for child in node.children:
            self._traverse_helper(child)

    def add_child(self, value):
        new_node = Node(value, father=self.current)
        self.current.add_child(new_node)
        self.current = new_node

    def get_ancestors(self, remove_root=True, child_first=False):
        ancestors = self.current.get_ancestors(
            remove_root=remove_root, child_first=child_first
        )
        return ancestors

    def set_current(self, node):
        self.current = node

    def is_root(self):
        return self.current == self.root

    def visualize(self, filename="tree"):
        
        plt.clf()
        graph = nx.DiGraph()

        # 构建图
        self._visualize_helper(self.root, graph)

        # 使用 Graphviz 的 dot 布局
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

        # 获取节点标签
        labels = {node: data["label"] for node, data in graph.nodes(data=True)}

        # 绘制图形
        plt.figure(figsize=(12, 8))
        nx.draw(
            graph, pos, labels=labels, with_labels=True,
            node_size=2000, node_color="skyblue",
            font_size=8, font_weight="bold", arrows=True
        )

        # 适应图形，避免被裁剪
        plt.tight_layout()
        plt.savefig(f"{filename}.png", bbox_inches="tight")
        plt.close()

    def _visualize_helper(self, node, graph, parent_id=None):
        node_id = str(id(node))
        
        # 这里指明可视化的信息
        action = str(node.value["action"].tool) if node.value["action"] else ""
        observation = str(node.value["observation"])[:80] if node.value["observation"] else ""
        # TODO 为什么有 There is an error....
        
        reward = f"{node.value['reward']:.3f}" if node.value["reward"] else ""
        label = f"{action}\n{observation}\n{reward}"
        
        graph.add_node(node_id, label=label)

        if parent_id:
            graph.add_edge(parent_id, node_id)
        for child in node.children:
            self._visualize_helper(child, graph, node_id)


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
        self.tree = MCSearchTree({"action": None, "observation": "", "reward": 0.0})
        self.examplesel = CustomExampleSelector() if self.use_example else None
        self.examples = ""
        self.total_step = 0
        self.parsing_error_alarm = "Failed to parse"
        self.good_base_reward = good_base_reward
        self.bad_base_reward = bad_base_reward
        self.decay_rate = decay_rate
        self.possible_anwsers = []

    def init_new_tree(self, video_name, question, possible_anwsers=[]):
        self.total_step = 0
        self.possible_anwsers = possible_anwsers
        self.video_name = video_name
        self.question = question
        self.tree = MCSearchTree({"action": None, "observation": "", "reward": 0.0})
        if self.use_example:
            self.examples = self.examplesel.select_examples(question)

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
        self.init_new_tree(video_name, question, possible_anwsers=possible_anwsers)
        anwsers = {"good_anwsers": [], "bad_anwsers": []}
        num_try = 0
        
        # 进入主循环的迭代
        while (
            len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) < max_answer
            and num_try < max_try
        ):

            print(f"\n\nTree {num_try}:\n\n") 
            
            ### visualize tree 
            self.tree.traverse()
            tree_filepath = f"viz/{quid}"
            if not os.path.exists(tree_filepath):
                os.makedirs(tree_filepath)
            tree_filename = f"viz/{quid}/tree_{num_try}"
            self.tree.visualize(filename=tree_filename)

            num_try += 1
            num_anwser = len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) + 1
            
            print(f"\n\nAnswer {num_anwser} - Try {num_try}:\n\n")
            
            # TODO 能不能整合简化一下 expansion 和 simulation
            # observation, is_good_result = self.expansion()  # 1-step expansion
            
            # 好观察才模拟, 坏观察不模拟
            # if is_good_result:
            #     answer, is_good_result = self.simulation()
            # else:
            #     answer = observation
            
            observation = ""
            answer = ""
            is_good_result = True
            
            ### expansion
            ancestor_history = self.get_ancestor_history()
            child_history = self.get_child_history()
            
            agent = self.get_new_agent(chain_history=ancestor_history, thought_prompt=child_history)
            agent_iterator = agent.iter(self.question)
            
            MAX_STEP = 10
            for step_idx, step in enumerate(agent_iterator):
                inter_step = step.get("intermediate_step")
                final_answer = step.get('output')
                
                if inter_step:
                    self.total_step += 1
                    action, observation = inter_step[0]
                    self.tree.add_child({"action": action, "observation": observation, "reward": 0.0})
                    
                    # if not self.is_good_observation(observation):
                    #     is_good_result = False
                    #     break
                if step_idx + 1 >= MAX_STEP:
                    break
            
            is_good_result = self.is_good_observation(observation)
                
            # 给叶节点设置 final_answer
            self.tree.current.value["final_anwser"] = final_answer
            
            
            # 为下一个循环挑选待扩展节点
            self.selection()
            
            if is_good_result:
                # 只有这里才会输出答案
                anwsers["good_anwsers"].append(answer)
            else:
                anwsers["bad_anwsers"].append(answer)

        return anwsers

    def get_new_agent(self, chain_history="", thought_prompt=""):
        template = self.template

        # 这里是送到 agent 里面的指令
        prefix, instruction, suffix = (
            template["prefix"].format(video_filename=self.video_name),
            template["format_instructions"],
            template["suffix"].format_map(
                SafeDict(
                    chain_history="\n" + chain_history if chain_history else "",
                    thought_prompt=thought_prompt,
                )
            ),
        )

        if self.use_example:
            suffix = (
                template["examples"].format_map(SafeDict(examples=self.examples))
                + suffix
            )

        # 可能需要改成特殊的 tool calling agent
        agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=self.parsing_error_alarm,
            agent_kwargs={
                "prefix": prefix,
                "format_instructions": instruction,
                "suffix": suffix,
            },
        )

        return agent

    def get_ancestor_history(self):
        tree_history = ""
        history_nodes = self.tree.get_ancestors(remove_root=True, child_first=False)
        history_nodes.append(self.tree.current)
        if not self.tree.is_root():
            for node in history_nodes:
                tree_history += "\n" + self.format_node_info(node)
        return tree_history

    def get_child_history(self):
        template = """ I have thought about the next action before, such as {tree_history}. I want to think out a different action. Regarding the now state and previous action candidates, I"""
        tree_history = ""
        history_nodes = self.tree.current.children
        if len(history_nodes) > 0:
            for node in history_nodes:
                if tree_history != "":
                    tree_history += ", "
                tree_history += f'"{node.value["action"].tool}" with Input "{node.value["action"].tool_input}" and Observation "{node.value["observation"]}"'
                # tree_history += f'"{node.value["action"].tool}" with input "{node.value["action"].tool_input}"'
            return template.format(tree_history=tree_history)
        else:
            return ""

    def format_node_info(self, node):
        action, observation = node.value["action"], node.value["observation"]
        return f"Thought: {action.log}\nObservation: {observation}"

    # TODO 考察是否应该 sample_all_expandable_nodes:
    def selection(self, sample_all_expandable_nodes=False):
        
        # 根据节点的奖励值（reward）来选择一个节点，并将其设置为当前节点
        
        # TODO 可以改写一下这边的 selection 策略
        
        if sample_all_expandable_nodes:
            all_descendant = self.tree.root.get_descendants(remove_leaf=True)
            nodes = all_descendant + [self.tree.root]
        else:
            ancestors = self.tree.get_ancestors(remove_root=False, child_first=False)
            nodes = ancestors
        
        # 随机选节点
        # TODO 根据 RAG 选节点
        node_sample = np.random.choice(nodes)
          
        self.tree.set_current(node_sample)

    def expansion(self, max_step=1):
        ancestor_history = self.get_ancestor_history()
        if ancestor_history != "":
            print("ancestor_history:", ancestor_history)
        child_history = self.get_child_history()
        if child_history != "":
            print("child_history:", child_history)
        agent = self.get_new_agent(
            chain_history=ancestor_history, thought_prompt=child_history
        )
        agent_iterator = agent.iter(self.question)

        observation = ""
        is_good_result = True
        
        
        for step_idx, step in enumerate(agent_iterator):
            # pdb.set_trace()
            if output := step.get("intermediate_step"):
                self.total_step += 1
                action, observation = output[0]
                self.tree.add_child(
                    {"action": action, "observation": observation, "reward": 0.0}
                )
                # 观察不好的话，就退出
                if not self.is_good_observation(observation):
                    is_good_result = False
                    break
            if step_idx + 1 >= max_step:
                break
        return observation, is_good_result

    def simulation(self):
        
        # 用于模拟从当前节点到终止状态的过程，并评估该路径的结果。
        # TODO 比较 expansion 和 simulation 的区别, 例如 simulation 不考虑 child_history
        ancestor_history = self.get_ancestor_history()
        
        child_history = self.get_child_history()
        print("simulation child_history:", child_history)
        assert child_history == ""
        print("simulation child_history done")
             
        agent = self.get_new_agent(chain_history=ancestor_history)
        agent_iterator = agent.iter(self.question)

        is_good_result = True
        inter_step = None
        final_answer = None
        
        
        for step in agent_iterator:
            print(f"\nagent_iterator step: {step}\n")
            # pdb.set_trace()
            # if output := step.get("intermediate_step"):
            inter_step = step.get("intermediate_step")  # TODO 这个 intermediate_step 是什么意思
            final_answer = step.get('output')
            if inter_step:
                self.total_step += 1
                action, observation = inter_step[0]
                self.tree.add_child(
                    {"action": action, "observation": observation, "reward": 0.0}
                )
                if not self.is_good_observation(observation):
                    is_good_result = False
                    break
        if not is_good_result:
            return observation, is_good_result

        # final_answer = agent_iterator.final_outputs["output"]
        self.tree.current.value["final_anwser"] = final_answer

        if not self.is_good_final_result(final_answer):
            is_good_result = False

        return final_answer, is_good_result

    def is_good_observation(self, observation):
        if self.parsing_error_alarm in observation:
            return False
        return True

    def is_good_final_result(self, final_result):
        if self.parsing_error_alarm in final_result:
            return False
        if len(self.possible_anwsers) > 0:
            for possible_anwser in self.possible_anwsers:
                if possible_anwser in final_result:
                    return True
            return False
        return True
    