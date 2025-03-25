import langchain
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from project.sql_template import (
    _sqlite_prompt,
    COUNTING_EXAMPLE_PROMPT,
    PROMPT_SUFFIX,
    TEMPORAL_EXAMPLE_PROMPT,
    REASONFINDER_ADDITION_PROMPT,
    HOWSEEKER_ADDITION_PROMPT,
    DESCRIP_EXAMPLE_PROMPT,
    DESCRIP_ADDITION_PROMPT,
)
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache

# 启用缓存 (SQLite 方式), 可以去掉这行代码进行对比
langchain.llm_cache = SQLiteCache(database_path="langchain_cache.db")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key='sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra',
    base_url="https://api.juheai.top/v1",  # OpenAI 的基础 URL
)

sql_path = '/share_data/NExT-QA/NExTVideo/1006/8968804598.db'

TOP_K = 2


sql_prompt = PromptTemplate(
    input_variables = ["input", "table_info", "top_k"],
    template = _sqlite_prompt + COUNTING_EXAMPLE_PROMPT+ PROMPT_SUFFIX,
)

db = SQLDatabase.from_uri(
    "sqlite:///" + sql_path, sample_rows_in_table_info=2
)


question = "Why is the blue sweater guy looking at the shirtless men?"


db_chain = SQLDatabaseChain.from_llm(
    llm=llm, db=db, top_k=TOP_K, verbose=True, prompt=sql_prompt
)

# try:
result = db_chain.run(question)
# except:
#     result ="There is an error. Try to ask the question in a different way."


 
'''
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
res = agent_executor.run(question)
print("SQL Query Result:")
print(res)
'''
