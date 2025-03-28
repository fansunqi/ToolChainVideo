import langchain
import os
import pickle
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pprint import pprint

# 设置缓存
# cache = SQLiteCache(database_path="langchain_cache.db")
# langchain.llm_cache = cache
# set_llm_cache(cache)

# 初始化 LLM
model = ChatOpenAI(
    api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
    model="gpt-4o",
    base_url="https://api.juheai.top/v1",
)

# 工具函数
@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

tools = [magic_function]

# 创建代理
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 缓存文件路径
cache_file = "gpt35_cache.pkl"

# 加载缓存
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

# 查询
query = "what is the value of magic_function(3)?"

# 检查缓存
if query in cache:
    print("Cache hit!")
    pprint(cache[query])
else:
    print("Cache miss. Calling API...")
    result = []
    for step in agent_executor.stream({"input": query}):
        pprint(step)
        result.append(step)
    cache[query] = result

    # 保存缓存
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)