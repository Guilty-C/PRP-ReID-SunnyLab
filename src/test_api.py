from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://usa.vimsai.com/v1",   # VimsAI 地址
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",   # 确认 VimsAI 支持的模型名
    messages=[{"role": "user", "content": "测试一下 API"}],
)

print(resp.choices[0].message.content)
