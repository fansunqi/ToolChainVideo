from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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

'''
### langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

ans = agent_executor.invoke({"input": query})
print(ans)
'''


### langgraph 省去了 scratchpad 的概念
from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools)

messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})

ans = messages["messages"][-1].content
print(ans)

# 再来一次
message_history = messages["messages"]
print(message_history)

new_query = "Pardon?"

messages = langgraph_agent_executor.invoke(
    {"messages": message_history + [("human", new_query)]}
)

print(messages["messages"][-1].content)