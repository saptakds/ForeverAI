#!/usr/bin/env python3
"""
Author: Saptak Das
Date: April 2025

Description:
    Diary PDF Semantic Indexer — Converts personal diary PDFs into searchable, 
    vectorized embeddings using FAISS and HuggingFace sentence transformers.

    Features:
    - PDF text extraction and cleaning
    - Date-based entry segmentation
    - Chunking of diary entries
    - SHA-256 deduplication
    - Vector embedding + FAISS index persistence
"""

import os
import re
import json
import hashlib
import numpy as np
import faiss
import fitz  # PyMuPDF
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --------------------- ⚙️ ENVIRONMENT CONFIG ---------------------
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------- 📁 FILE PATHS ---------------------
PDF_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/Diary Content.pdf"
FAISS_INDEX_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/diary_faiss.index"
CHUNK_META_PATH = "/Users/saptakds/Documents/WIP Projects/Forever AI/py_forever_ai/datasource/chunk_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------- 🔥 EMBEDDING MODEL INIT ---------------------
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")

# --------------------- 📄 PDF TEXT EXTRACTION ---------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

# --------------------- 🧹 TEXT CLEANING ---------------------
def clean_text(raw_text: str) -> str:
    return " ".join(raw_text.replace("\n", " ").split())

# --------------------- 📆 DATE-BASED SPLITTING ---------------------
def split_text_by_date(text: str):
    date_pattern = r"(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b)"
    entries = re.split(date_pattern, text)
    return [{"date": entries[i].strip(), "text": entries[i + 1].strip()} for i in range(1, len(entries), 2)]

# --------------------- 📦 TEXT CHUNKING ---------------------
def split_into_chunks(text: str, max_chunk_size: int = 500):
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

# --------------------- 🔒 CONTENT HASHING ---------------------
def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# --------------------- 📂 METADATA MANAGEMENT ---------------------
def load_metadata() -> dict:
    if os.path.exists(CHUNK_META_PATH):
        with open(CHUNK_META_PATH, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata: dict) -> None:
    with open(CHUNK_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

# --------------------- 🧠 FAISS INDEX MANAGEMENT ---------------------
def load_faiss(dim: int = 384):
    return faiss.read_index(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else faiss.IndexFlatL2(dim)

def save_faiss(index) -> None:
    faiss.write_index(index, FAISS_INDEX_PATH)

# --------------------- 🚀 MAIN INDEXING LOGIC ---------------------
def index_diary():
    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    entries = split_text_by_date(cleaned_text)

    existing_meta = load_metadata()
    faiss_index = load_faiss()

    new_vectors = []
    new_ids = []
    updated_meta = existing_meta.copy()
    next_id = max(map(int, existing_meta.keys()), default=-1) + 1

    for entry in entries:
        chunks = split_into_chunks(entry["text"])
        for chunk in chunks:
            chunk_hash = hash_chunk(chunk)

            # Deduplication check
            if chunk_hash in (meta["hash"] for meta in existing_meta.values()):
                continue

            embedding = embedder.get_text_embedding(f"{entry['date']}: {chunk}")
            new_vectors.append(np.array(embedding, dtype=np.float32))
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

# --------------------- 🏁 SCRIPT ENTRYPOINT ---------------------
if __name__ == "__main__":
    index_diary()
