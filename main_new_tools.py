import argparse
from omegaconf import OmegaConf

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
from tools.frame_selector import select_frames




def tool_chain_reasoning( 
    input_question, 
    llm, 
    tools,
    recursion_limit=24,  
    use_cache=True,
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
    parser.add_argument('--config', default="config/nextqa.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)
    
    # 视频路径
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    question_w_options = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
    
    tool_planner_llm = ChatOpenAI(
        api_key = conf.openai.GPT_API_KEY,
        model = conf.openai.GPT_MODEL_NAME,
        temperature = 0,
        base_url = conf.openai.PROXY
    )

    # 初始化 YOLO 模型
    yolo_model_path = "checkpoints/yoloe-11l-seg.pt"
    yolo_tracker = YOLOTracker(yolo_model_path)

    tool_instances = [yolo_tracker]

    # 先搞一个 uniform frame selector
    video_stride = 30  # 设置视频 stride，跳过的帧数
    frames = select_frames(video_path=video_path, video_stride=video_stride)
    
    for tool_instance in tool_instances:
        tool_instance.set_frames(frames)

    tools = []
    for tool_instance in tool_instances:
        for e in dir(tool_instance):
            if e.startswith("inference"):
                func = getattr(tool_instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))

    tool_chain_output = tool_chain_reasoning(
        input_question=question_w_options,
        llm=tool_planner_llm,
        tools=tools
    )

    print(tool_chain_output)



    