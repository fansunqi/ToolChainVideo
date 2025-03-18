from langchain_community.llms import OpenAI

llm = OpenAI(
    openai_api_key = self.config.openai.GPT_API_KEY,
    model_name = self.config.openai.GPT_MODEL_NAME,
    temperature = 0,
    openai_api_base = self.config.openai.PROXY
)


a = llm("hello, i am a human")
print(a)
b = chat("hello, i am a human")
print(b)