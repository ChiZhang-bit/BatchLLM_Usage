import tiktoken
from openai import OpenAI

client = OpenAI(api_key="sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c", base_url="https://cd.aiskt.com/v1")

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "test"}
  ]
)

print(response.choices[0].message)
encoding = tiktoken.get_encoding("cl100k_base")
print(len(encoding.encode("test")))
