import fitz  # PyMuPDF for PDF text extraction
import re
import torch
import os
import json
import numpy as np
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_cpp import Llama

# ---- 🛠️ Fix Out of Memory Issues ----
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- 🔥 Load Embedding Model ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")

# ---- 🚀 Load Llama 3 Model ----
LLAMA_MODEL_PATH = "/Users/saptakds/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf"
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=6)

# ---- 📜 Extract Text from PDF ----
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc]).strip()

# ---- 🧹 Clean Text ----
def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

# ---- 📆 Split Text by Date ----
def split_text_by_date(text):
    date_pattern = r"(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b)"
    entries = re.split(date_pattern, text)
    diary_entries = []

    for i in range(1, len(entries), 2):
        date = entries[i].strip()
        entry_text = entries[i + 1].strip() if i + 1 < len(entries) else ""
        diary_entries.append({"date": date, "text": entry_text})

    return diary_entries

# ---- ✂️ Split into Meaningful Chunks ----
def split_into_meaningful_chunks(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ---- 🧠 Generate Embeddings ----
def generate_embeddings(diary_chunks):
    embedded_entries = []

    for entry in diary_chunks:
        text = f"{entry['date']}: {entry['chunk_text']}"
        embedding = embedder.get_text_embedding(text)
        embedded_entries.append({
            "date": entry["date"],
            "chunk_text": entry["chunk_text"],
            "embedding": embedding
        })

    return embedded_entries

# ---- 💾 Save to FAISS Index ----
def save_faiss_index(embedded_entries, index_path="diary_index.faiss", meta_path="diary_meta.json"):
    dimension = len(embedded_entries[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    vectors = [entry["embedding"] for entry in embedded_entries]
    metadata = [{"date": entry["date"], "chunk_text": entry["chunk_text"]} for entry in embedded_entries]

    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

# ---- 📤 Load FAISS Index ----
def load_faiss_index(index_path="diary_index.faiss", meta_path="diary_meta.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return index, metadata

# ---- 🔍 Find Best Match Using FAISS ----
def find_best_match_faiss(query, index, metadata):
    query_vector = np.array([embedder.get_text_embedding(query)]).astype("float32")
    D, I = index.search(query_vector, k=1)
    if len(I[0]) == 0 or I[0][0] == -1:
        return None
    return metadata[I[0][0]]

# ---- 📝 Generate AI Response ----
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

    Diary Entry: "{best_entry['chunk_text']}"  
    Question: "{query}" 
    
    Response:
    """

    try:
        output = llm(prompt, max_tokens=200)
    except Exception as e:
        print(f"❌ Error: Llama call failed -> {e}")
        return "Error communicating with Llama."

    if "choices" in output and output["choices"]:
        return output["choices"][0]["text"].strip()
    return "No response from the model."

# ---- 🏁 MAIN SCRIPT ----
if __name__ == "__main__":
    index_path = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_index.faiss"
    meta_path = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_meta.json"
    pdf_path = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/Diary Content.pdf"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("🔧 First run: Extracting and indexing data...")
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        diary_entries = split_text_by_date(cleaned_text)

        final_chunks = []
        for entry in diary_entries:
            sub_chunks = split_into_meaningful_chunks(entry["text"])
            for chunk in sub_chunks:
                final_chunks.append({"date": entry["date"], "chunk_text": chunk})

        embedded_data = generate_embeddings(final_chunks)
        save_faiss_index(embedded_data, index_path, meta_path)

        print("✅ Index created and saved!")
    else:
        print("✅ FAISS index loaded from disk.")

    faiss_index, meta_data = load_faiss_index(index_path, meta_path)

    while True:
        query = input("\n🔍 Enter your search query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        best_entry = find_best_match_faiss(query, faiss_index, meta_data)
        response = generate_response(query, best_entry)
        print(f"\n💬 AI Response: {response}")
