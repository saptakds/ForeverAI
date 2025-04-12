#!/usr/bin/env python3
"""
Author: Saptak Das
Date: April 2025
Description:
    AI-Powered Digital Memory — A hybrid setup for semantic recall from diary data,
    powered by FAISS vector search and LLM-backed conversational responses.
    
    Supports:  
    - Local inference with Meta Llama 3.1 8B (via llama.cpp)  
    - Cloud inference with Azure OpenAI GPT-4o  

    Easily switch modes with the `USE_LOCAL_LLM` flag.
"""

import os
import torch
import json
import numpy as np
import faiss
import tiktoken
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --------------------- 💡 CONFIGURATION ---------------------
USE_LOCAL_LLM = False  # Set to False for Azure GPT-4o

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_faiss.index"
CHUNK_META_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/chunk_metadata.json"

# --------------------- 🧹 ENVIRONMENT SETUP ---------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------- 🔥 EMBEDDING MODEL ---------------------
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")

# --------------------- 🧠 TOKEN PROFILER ---------------------
def count_tokens(prompt: str, model_name="gpt-4"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(prompt))

# --------------------- 🗄️ FAISS INDEX LOAD ---------------------
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNK_META_PATH, "r") as meta_file:
    metadata = json.load(meta_file)

# --------------------- ⚡ LLM INITIALIZATION ---------------------
if USE_LOCAL_LLM:
    from llama_cpp import Llama
    LLAMA_MODEL_PATH = "/Users/saptakds/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf"
    llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=6)
else:
    from openai import AzureOpenAI
    AZURE_OPENAI_KEY = "xxxxxx"
    AZURE_OPENAI_ENDPOINT = "https://azureopenai-forever-ai.openai.azure.com/"
    AZURE_DEPLOYMENT_NAME = "gpt-4o"

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

# --------------------- 🔍 SEMANTIC SEARCH ---------------------
def find_best_match(query: str):
    embedding = embedder.get_text_embedding(query)
    distances, indices = faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
    index_str = str(indices[0][0])
    return metadata.get(index_str), distances[0][0]

# --------------------- 💬 RESPONSE GENERATION ---------------------
def generate_response(query: str, best_entry: dict) -> str:
    if not best_entry:
        return "I don't remember anything related to that."

    if USE_LOCAL_LLM:
        # Local Llama Prompt
        prompt = f"""
        You are a digital memory of a person, created from their diary.
        Respond naturally as if you are recalling a past experience.
        Keep it under 200 tokens.

        Diary Entry: "{best_entry['text']}"
        Question: "{query}"
        Response:
        """
        try:
            output = llm(prompt, max_tokens=100)
            return output["choices"][0]["text"].strip() if output.get("choices") else "No response from the model."
        except Exception as e:
            return f"❌ Llama Error: {e}"

    else:
        # Azure GPT Prompt
        system_prompt = "You are the user's memory. Respond in first person, naturally recalling a past experience."
        user_prompt = f"""
        Diary Entry: "{best_entry['text']}"
        Question: "{query}"
        Respond as if you remember it personally.
        """
        total_tokens = count_tokens(system_prompt + user_prompt)
        print(f"\n📊 Token Profiler")
        print(f"- System Prompt: {count_tokens(system_prompt)} tokens")
        print(f"- Diary Entry: {count_tokens(best_entry['text'])} tokens")
        print(f"- Question: {count_tokens(query)} tokens")
        print(f"- Total Prompt Tokens: {total_tokens}")
        print(f"- Estimated Max Completion: 100")
        print(f"- Total Estimate (input + output): {total_tokens + 100}")

        try:
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Azure LLM Error: {e}"

# --------------------- 🏁 CLI APPLICATION ---------------------
def main():
    print(f"🧠 Memory Mode: {'Local Llama' if USE_LOCAL_LLM else 'Azure GPT-4o'}\n")
    while True:
        query = input("\n🔍 Ask your friend: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        best_entry, _ = find_best_match(query)
        answer = generate_response(query, best_entry)
        print(f"\n💬 Response: {answer}")

if __name__ == "__main__":
    main()
