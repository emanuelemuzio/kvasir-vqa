import ollama
import os
from dotenv import load_dotenv

load_dotenv()

response = ollama.chat(model=os.getenv('CHAIN_OF_THOUGHT_MODEL'), messages=[
    {
        "role": "system",
        "content": "You are a medical expert. Reasonate about what the focus of the question is."
    },
    {
        "role": "user",
        "content": "Have all polyps been removed?"
    },
])
print(response['message']['content'])