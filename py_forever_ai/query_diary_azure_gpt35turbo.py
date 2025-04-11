import os
import torch
import json
import numpy as np
import faiss
from openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---- 🛠️ Fix Out of Memory Issues ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- ⚙️ Azure OpenAI Config ----
AZURE_OPENAI_KEY = "xxxxx"
AZURE_OPENAI_ENDPOINT = "https://sdas2-m9ckxlb3-eastus.cognitiveservices.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"  # 👈 Use your deployed model name

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---- 📁 Paths ----
FAISS_INDEX_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_faiss.index"
CHUNK_META_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/chunk_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- 🔥 Load Embedding Model ----
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")

# ---- 🔎 Load FAISS Index + Metadata ----
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNK_META_PATH, "r") as f:
    metadata = json.load(f)

# ---- 🔍 Semantic Search ----
def find_best_match(query):
    emb = embedder.get_text_embedding(query)
    D, I = faiss_index.search(np.array([emb], dtype=np.float32), k=1)
    idx = str(I[0][0])
    return metadata.get(idx), D[0][0]

# ---- 🤖 Azure LLM Response ----
def generate_response(query, best_entry):
    if not best_entry:
        return "I don't remember anything related to that."

    prompt = f"""
    You are a digital memory of a person, created from their diary.  
    When answering, respond naturally as if you're recalling a past experience, not reading from a log.
    Try to wrap up your thought naturally, and keep your response under 200 tokens.

    - Speak in first person, as if you are remembering the moment.  
    - Don't mention "diary" or "entries." Just answer as yourself.  
    - If details are vague, respond as a person would ("I don’t remember much, but I think...").  
    - If something is unclear, acknowledge it rather than guessing.  
    - Finish your thought. Avoid trailing off or stopping mid-sentence.  

    Diary Entry: "{best_entry['text']}"  
    Question: "{query}" 

    Response:
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a memory-based assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Azure LLM Error: {e}"

# ---- 🏁 CLI App ----
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask your memory: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        best_entry, _ = find_best_match(query)
        answer = generate_response(query, best_entry)
        print(f"\n💬 Response: {answer}")
