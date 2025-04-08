

import pickle

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
)

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent




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
        
        step_idx += 1
        steps.append(step)
 
    try:
        output = steps[-1]["messages"][-1].content
    except:
        output = None
    
    print(f"\nToolChainOutput: {output}") 
    return output


if __name__ == "__main__":

    video_stride = 30
    
    