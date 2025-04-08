

import pickle

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
)

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent


mannual_cache = None
mannual_cache_file = None


def ToolChainReasoning( 
    input_question, 
    llm, 
    tools,
    recursion_limit,  
    use_cache=True,
):
    # TODO 考虑之前选择工具的历史
    
    # TODO 研究这个 prompt
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
    
    # tool_planner_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful video analysis assistant equipped with various tools"),
    #     ("placeholder", "{messages}"),
    #     ("user", "Remember, always be polite!"),
    # ])
    
    # tool_planner_prompt = "You are a helpful video analysis assistant equipped with various tools."
    
    # TODO 这里加上 system_prompt，告诉工具与使用工具的规则
    tool_planner = create_react_agent(llm, tools, state_modifier=_modify_state_messages)
    
    query = QUERY_PREFIX + input_question + '\n\n' + TOOLS_RULE
    
    if use_cache and (query in mannual_cache):
        print("\nCache hit!")
        steps = mannual_cache[query]
    else:
        print("\nCache miss. Calling API...")
        steps = []
        global step_idx
        step_idx = 0
        
        # TODO 研究一下这里的 stream_mode
        for step in tool_planner.stream(
            {"messages": [("human", query)]}, 
            {"recursion_limit": recursion_limit},
             stream_mode="values"):
            # stream_mode="updates"):
            
            # pdb.set_trace()
            step_message = step["messages"][-1]
            # 打印 step_message
            if isinstance(step_message, tuple):
                print(step_message)
            else:
                step_message.pretty_print()
            # pdb.set_trace()
            
            step_idx += 1
            steps.append(step)
                
        mannual_cache[query] = steps
        print("\nSaving cache...")
        with open(mannual_cache_file, "wb") as f:
            pickle.dump(mannual_cache, f)
 
    try:
        # updates mode
        # output = steps[-1]['agent']['messages'][0].content
        # value mode
        output = steps[-1]["messages"][-1].content
    except:
        output = None
    
    print(f"\nToolChainOutput: {output}") 
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/nextqa.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    seed_everything(conf.seed) 
    
    # 将 main.py 文件自身和 opt.config 文件复制一份存储至 conf.output_path
    current_script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    shutil.copy(current_script_path, os.path.join(conf.output_path, f"main_{timestamp}.py"))
    # 复制 opt.config 文件到输出目录
    config_basename = os.path.basename(opt.config).split('.')[0]
    shutil.copy(opt.config, os.path.join(conf.output_path, f"{config_basename}_{timestamp}.yaml"))
    
    if TO_TXT:
        # 指定输出 log
        log_path = os.path.join(conf.output_path, f"log_{timestamp}.txt")
        f = open(log_path, "w")
        # 重定向标准输出
        sys.stdout = f
    
    # mannual LLM cache
    mannual_cache_file = conf.mannual_cache_file
    if os.path.exists(mannual_cache_file):
        print(f"\nLoading LLM cache from {mannual_cache_file}...")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"\nCreating LLM cache: {mannual_cache_file}...")
        mannual_cache = {}