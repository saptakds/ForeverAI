import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_cpp import Llama

# ---- Paths ----
FAISS_INDEX_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_faiss.index"
CHUNK_META_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/chunk_metadata.json"
LLAMA_MODEL_PATH = "/Users/saptakds/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf"

# ---- Load Stuff ----
embedder = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=6)

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNK_META_PATH, "r") as f:
    metadata = json.load(f)

# ---- Query Flow ----
def find_best_match(query):
    emb = embedder.get_text_embedding(query)
    D, I = faiss_index.search(np.array([emb], dtype=np.float32), k=1)
    idx = str(I[0][0])
    return metadata.get(idx), D[0][0]

def generate_response(query, best_entry):
    if not best_entry:
        return "I don't remember anything related to that."

    prompt = f"""
    You are a digital memory of a person, created from their diary.  
    When answering, respond **naturally** as if you are recalling a past experience, not reading from a log.
    Keep your answer under 200 tokens. 

    - **Speak in first person**, as if you are remembering the moment.  
    - **Don't mention "diary" or "entries."** Just answer as yourself.  
    - **If details are vague, respond as a person would** ("I don’t remember much, but I think...").  
    - **If something is unclear, acknowledge it rather than guessing.**  

    Diary Entry: "{best_entry['text']}"  
    Question: "{query}" 
    
    Response:
    """
    output = llm(prompt, max_tokens=200)
    return output["choices"][0]["text"].strip() if "choices" in output else "No response from model."

# ---- Run ----
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask your memory: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        best_entry, _ = find_best_match(query)
        answer = generate_response(query, best_entry)
        print(f"\n💬 Response: {answer}")
