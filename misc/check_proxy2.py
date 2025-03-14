from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo",  # 或其他您选择的模型
    temperature=0.7,
    max_tokens=150,
    api_key='sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra',
    base_url="https://api.juheai.top/v1"  # OpenAI 的基础 URL
)

messages = [
    SystemMessage(content="你是一名友好的聊天机器人。"),
    HumanMessage(content="你好，能和我聊聊天气吗？")
]

response = chat.invoke(messages)
print(response.content)
