import os
import torch
import json
import numpy as np
import faiss
import tiktoken  # 🆕 For token counting
from openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---- 🛠️ Fix Out of Memory Issues ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- ⚙️ Azure OpenAI Config ----
AZURE_OPENAI_KEY = "xxxxxxx"
AZURE_OPENAI_ENDPOINT = "https://azureopenai-forever-ai.openai.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o"

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

# ---- 🧠 Token Profiler ----
def count_tokens_gpt(prompt: str, model=AZURE_DEPLOYMENT_NAME):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(prompt))

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

    diary_entry = best_entry['text']
    system_prompt = "You are the user's memory. Respond in first person, naturally recalling a past experience."

    user_prompt = f"""
    Diary Entry: "{diary_entry}"
    Question: "{query}"
    Respond as if you remember it personally.
    """

    # --- 🧠 Token Debug Info ---
    input_tokens = count_tokens_gpt(system_prompt + user_prompt)
    diary_tokens = count_tokens_gpt(diary_entry)
    question_tokens = count_tokens_gpt(query)

    print(f"\n📊 Token Profiler")
    print(f"- System Prompt: {count_tokens_gpt(system_prompt)} tokens")
    print(f"- Diary Entry: {diary_tokens} tokens")
    print(f"- Question: {question_tokens} tokens")
    print(f"- Total Prompt Tokens: {input_tokens}")
    print(f"- Estimated Max Completion: 200")
    print(f"- Total Estimate (input + output): {input_tokens + 200}")

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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
