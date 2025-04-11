import fitz
import os
import re
import json
import hashlib
import numpy as np
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.listdir

# ---- Paths ----
PDF_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/Diary Content.pdf"
FAISS_INDEX_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_faiss.index"
CHUNK_META_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/chunk_metadata.json"

# ---- Load Embedder ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")

# ---- PDF & Text Preprocessing ----
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

def split_text_by_date(text):
    date_pattern = r"(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|...|Dec)[a-z]* \d{1,2}, \d{4})\b)"
    entries = re.split(date_pattern, text)
    return [{"date": entries[i].strip(), "text": entries[i + 1].strip()} for i in range(1, len(entries), 2)]

def split_into_chunks(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []
    for sentence in sentences:
        if sum(len(s) for s in current) + len(sentence) <= max_chunk_size:
            current.append(sentence)
        else:
            chunks.append(" ".join(current))
            current = [sentence]
    if current: chunks.append(" ".join(current))
    return chunks

# ---- Hashing ----
def hash_chunk(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ---- Load or Init Metadata ----
def load_metadata():
    if os.path.exists(CHUNK_META_PATH):
        with open(CHUNK_META_PATH, "r") as f:
            return json.load(f)
    return {}

def save_metadata(meta):
    with open(CHUNK_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# ---- Load or Init FAISS ----
def load_faiss(dim=384):
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return faiss.IndexFlatL2(dim)

def save_faiss(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

# ---- Main ----
if __name__ == "__main__":
    text = clean_text(extract_text_from_pdf(PDF_PATH))
    entries = split_text_by_date(text)

    existing_meta = load_metadata()
    faiss_index = load_faiss()

    new_vectors, new_ids = [], []
    updated_meta = existing_meta.copy()
    next_id = max(map(int, existing_meta.keys())) + 1 if existing_meta else 0

    for entry in entries:
        chunks = split_into_chunks(entry["text"])
        for chunk in chunks:
            chunk_hash = hash_chunk(chunk)
            if chunk_hash in (m["hash"] for m in existing_meta.values()):
                continue
            emb = embedder.get_text_embedding(f"{entry['date']}: {chunk}")
            new_vectors.append(np.array(emb, dtype=np.float32))
            updated_meta[str(next_id)] = {"date": entry["date"], "text": chunk, "hash": chunk_hash}
            new_ids.append(next_id)
            next_id += 1

    if new_vectors:
        faiss_index.add(np.array(new_vectors))
        save_faiss(faiss_index)
        save_metadata(updated_meta)
        print(f"✅ Added {len(new_vectors)} new chunks.")
    else:
        print("ℹ️ No new content to index.")
