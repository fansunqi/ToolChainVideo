import os
from openai import OpenAI
from pydantic import BaseModel

# 设置 OpenAI API 密钥
client = OpenAI(
    api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
    base_url="https://api.juheai.top/v1/"
)


class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]

response_format = PatchZoomerResponse.model_json_schema()
print(response_format)

# 创建聊天补全请求
# response = client.chat.completions.create(
#     model="gpt-4o",  # 确保使用支持结构化输出的模型
#     messages=[
#         {"role": "user", "content": "请提供一个用户信息，以 JSON 格式返回，包含姓名、年龄和爱好。"}
#     ],
#     response_format={
#         "type": "json_object",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string"},
#                 "age": {"type": "integer"},
#                 "hobbies": {
#                     "type": "array",
#                     "items": {"type": "string"}
#                 }
#             },
#             "required": ["name", "age", "hobbies"]
#         }
#     }
# )

# 输出结果
# print(response.choices[0].message.content)
