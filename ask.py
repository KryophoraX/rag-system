import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key
load_dotenv()

# Load the same embedding model used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# NVIDIA NIM
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.2,
    top_p=0.7,
    max_completion_tokens=1024,
)

# Ask a question
question = input("Ask a question: ")

# Retrieve relevant chunks
results = db.similarity_search(question, k=3)

# Combine retrieved text
context = "\n\n".join(
    doc.page_content for doc in results
)

# Give context to the LLM
prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information in the provided context.
If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{question}
"""

# Generate answer
response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)

# Show sources
print("\nSources:")
for doc in results:
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", 0) + 1
    print(f"- {source}, page {page}")