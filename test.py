from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("HF_TOKEN"))

from config import (
    THESIS_LLM_REPO_ID,
)

llm = HuggingFaceEndpoint(
    repo_id=THESIS_LLM_REPO_ID,
    # task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is your transformer architecture like? Explain in detail.")

print(result.content)