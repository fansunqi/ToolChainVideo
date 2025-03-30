from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pprint import pprint

model = ChatOpenAI(
    api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
    model="gpt-4o",
    base_url="https://api.juheai.top/v1",
    )


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"




### langgraph iteration
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)


def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"]}).to_messages()


app = create_react_agent(model, tools, state_modifier=_modify_state_messages)

for step in app.stream({"messages": [("human", query)]}, stream_mode="updates"):
    pprint(step)
    try:
        output = step['agent']['messages'][0].content
    except:
        output = None
   
# output = app.stream({"messages": [("human", query)]}, stream_mode="updates") 
print(output)
