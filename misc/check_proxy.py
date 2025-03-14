# from langchain_community.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import ChatMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key = "sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra", 
    model_name = "gpt-3.5-turbo", 
    openai_api_base = "https://api.juheai.top/v1", 
    temperature = 0
)

# llm = ChatOpenAI(
#     openai_api_key = "sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra", 
#     model_name = "gpt-3.5-turbo", 
#     openai_api_base = "https://api.juheai.top/v1", 
#     temperature = 0
# )

# res = llm("Hello, how are you?")
# print(res)

question = 'Inpaint the man on the right.'
flag_ref_vos = llm(f"Please determine if this task is related to inpaint, generally referring to the word inpaint in the task. Reply 0 if it is related and 1 if it is not. The task is as follows:{question}.")
print(flag_ref_vos)
# 创建 HumanMessage 对象
# message = HumanMessage(content="Hello, how are you?")

# # 调用 llm 对象
# res = llm([message])
# print(res)