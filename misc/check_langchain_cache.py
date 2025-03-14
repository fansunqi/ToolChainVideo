from langchain_community.cache import SQLiteCache
# from langchain_openai import OpenAI
# from langchain import OpenAI
from langchain_core.globals import set_llm_cache
from langchain.llms.openai import OpenAI


# 初始化慢速但稳定的模型示例，便于观察缓存的效果
llm = OpenAI(model_name="gpt-3.5-turbo", 
             openai_api_key = "sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
             openai_api_base = "https://api.juheai.top/v1",
             n=2, 
             best_of=2)


# 设置 SQLite 缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 第一次调用
print(llm("Tell me a joke"))

# 第二次调用，命中缓存
print(llm("Tell me a joke"))
