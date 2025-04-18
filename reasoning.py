import pdb

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
    ASSISTANT_ROLE,
)

from util import load_cache, save_cache

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent

from tools.yolo_tracker import YOLOTracker
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector
from tools.image_qa import ImageQA
from tools.temporal_grounding import TemporalGrounding


def spatiotemporal_reasoning(
    question,
    question_w_options,
    llm,
    tools, 
    recursion_limit=24,  
    use_cache=True,
    mannual_cache=None,
    mannual_cache_file=None
):

    for tool in tools:
        if isinstance(tool, TemporalGrounding):
            temporal_grounding = tool

    # 1. temporal grounding
    temporal_grounding_result = temporal_grounding.inference(input=question)

    # 2.1 image grid qa
    pdb.set_trace()

    # 2.2 时间截取送到 temporal qa 或者 video qa 中去

    # 2.3 frame selector + 空间工具，再处理

    # pdb.set_trace()






def langgraph_reasoning( 
    input_question, 
    llm, 
    tools,
    recursion_limit=24,  
    use_cache=True,
    mannual_cache=None,
    mannual_cache_file=None
):

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