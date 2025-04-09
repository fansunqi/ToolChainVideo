from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI



model = ChatOpenAI(
    api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
    model="gpt-4o",
    base_url="https://api.juheai.top/v1",
)


class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: int = Field(description="How funny the joke is, from 1 to 10")


class JokeList(BaseModel):
    """A list of jokes."""
    jokes: List[Joke] = Field(description="A list of jokes")


# 使用结构化输出生成多个笑话
structured_llm = model.with_structured_output(JokeList)
output = structured_llm.invoke("Tell me 3 jokes about cats")

# 打印结果
print("Generated Jokes:")
for i, joke in enumerate(output.jokes, 1):
    print(f"\nJoke #{i}:")
    print(f"Setup: {joke.setup}")
    print(f"Punchline: {joke.punchline}")
    print(f"Rating: {joke.rating}/10")

# 如果需要原始字典列表格式
jokes_dict_list = [joke.model_dump() for joke in output.jokes]
print("\nJokes as dictionary list:")
print(jokes_dict_list)