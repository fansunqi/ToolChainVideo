import langchain
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")
set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pprint import pprint



model = ChatOpenAI(
    api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
    model="gpt-4o",
    base_url="https://api.juheai.top/v1",
)


query = "what is the value of magic_function(3)?"



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for step in agent_executor.stream({"input": query}):
    pprint(step)