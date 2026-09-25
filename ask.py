import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["NVIDIA_API_KEY"]
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=api_key,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)