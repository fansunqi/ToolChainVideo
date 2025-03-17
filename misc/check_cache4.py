# pip install -U langchain langchain-openai langchain_community
import langchain
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 启用缓存 (SQLite 方式), 可以去掉这行代码进行对比
langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")

# 初始化 OpenAI Chat 模型
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=2.0,
    api_key='sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra',
    base_url="https://api.juheai.top/v1",  # OpenAI 的基础 URL
)

# 定义对话消息
# messages = [
#     SystemMessage(content="You are a friendly chatbot."),
#     HumanMessage(content="hello, can you chat with me?")
# ]

# # 进行对话
# response = chat.invoke(messages)
# print(response.content)

# # 再次进行对话
# response = chat.invoke(messages)
# print(response.content)

a = chat("hello, i am a human")
print(a)
b = chat("hello, i am a human")
print(b)
