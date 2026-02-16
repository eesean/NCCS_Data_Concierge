import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
  },
  data=json.dumps({
    "model": "deepseek/deepseek-r1-0528:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

data = response.json()
if "choices" in data:
    print(data["choices"][0]["message"]["content"])
else:
    print("API Error:", data)