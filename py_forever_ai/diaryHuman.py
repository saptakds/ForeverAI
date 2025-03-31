import fitz  # PyMuPDF for PDF text extraction
import re
import torch
import os
import json
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_cpp import Llama  # Replaces ctransformers

# ---- 🛠️ Fix Out of Memory Issues ----
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Remove MPS limit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA if on Mac

# ---- 🔥 Load Embedding Model (Smaller & Efficient) ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller than instructor-xl
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device="cpu")  # Force CPU

# ---- 🚀 Load Llama 3 Model (Using llama-cpp-python) ----
LLAMA_MODEL_PATH = "/Users/saptakds/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf"
# llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, temperature=0.7, top_k=50, top_p=0.95)

llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, n_threads=6)


# ---- 📜 Extract Text from PDF ----
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

# ---- 🧹 Clean Text ----
def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

# ---- 📆 Split Text by Date-Based Entries ----
def split_text_by_date(text):
    date_pattern = r"(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b)"
    entries = re.split(date_pattern, text)
    
    diary_entries = []
    for i in range(1, len(entries), 2):
        date = entries[i].strip()
        entry_text = entries[i + 1].strip() if i + 1 < len(entries) else ""
        diary_entries.append({"date": date, "text": entry_text})

    return diary_entries

# ---- ✂️ Split Large Entries into Chunks ----
def split_into_meaningful_chunks(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_embeddings(diary_chunks, batch_size=8):
    embedded_entries = []

    for i in range(0, len(diary_chunks), batch_size):
        batch = diary_chunks[i:i + batch_size]

        # Ensure batch is a list of strings
        texts = [f"{entry['date']}: {entry['chunk_text']}" for entry in batch]

        # 🔥 Fix: Call embedding function one by one instead of batch
        embedding_vectors = [embedder.get_text_embedding(text) for text in texts]

        for entry, embedding_vector in zip(batch, embedding_vectors):
            embedded_entries.append({
                "date": entry["date"],
                "chunk_text": entry["chunk_text"],
                "embedding": embedding_vector  # Convert NumPy array to list
            })

    return embedded_entries

# ---- 💾 Load Embeddings from JSON File ----
def load_embeddings(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# ---- 🔍 Find Best Matching Diary Entry (Cosine Similarity) ----
def find_best_match(query, embedded_data):
    query_embedding = embedder.get_text_embedding(query)
    
    best_match = None
    best_score = -1
    
    for entry in embedded_data:
        entry_embedding = np.array(entry["embedding"])
        score = np.dot(query_embedding, entry_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding))  # Cosine similarity

        if score > best_score:
            best_score = score
            best_match = entry

    return best_match, best_score

# ---- 📝 Generate AI Response Using Llama 3 ----
def generate_response(query, best_entry):
    if not best_entry:
        return "I don't remember anything related to that."

    prompt = f"""
    You are a digital memory of a person, created from their diary.
    Answer the following question based **only** on the provided diary entry.
    If the diary does not contain relevant information, say "I don't know."

    Diary Entry: "{best_entry['chunk_text']}"
    Question: "{query}"
    
    Response:
    """

    print("🛠️ Debug: Sending prompt to Llama...")  # ✅ Ensure it reaches here

    try:
        output = llm(prompt, max_tokens=50)
        print("🛠️ Debug: Llama Output Received")  # ✅ If this prints, Llama responded

    except Exception as e:
        print(f"❌ Error: Llama call failed -> {e}")
        return "Error communicating with Llama."

    print("🛠️ Debug: Full Output ->", output)  # ✅ See the full response

    if "choices" in output and output["choices"]:
        return output["choices"][0]["text"].strip()
    else:
        return "No response from the model."

# ---- 🏁 MAIN SCRIPT ----
if __name__ == "__main__":
    pdf_path = "/Users/saptakds/Documents/WIP Projects/Forever AI/Diary Content.pdf"

    # Extract, clean, and split text
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    diary_entries = split_text_by_date(cleaned_text)

    # Apply semantic chunking
    final_chunks = []
    for entry in diary_entries:
        sub_chunks = split_into_meaningful_chunks(entry["text"])
        for chunk in sub_chunks:
            final_chunks.append({"date": entry["date"], "chunk_text": chunk})

    print(f"✅ Extracted & chunked {len(final_chunks)} entries.")

    # Generate embeddings & save to JSON
    embedded_data = generate_embeddings(final_chunks)
    with open("diary_embeddings.json", "w") as f:
        json.dump(embedded_data, f, indent=4)

    print("✅ Embeddings generated & saved successfully!")

    # Load embeddings & start search loop
    embedded_data = load_embeddings("diary_embeddings.json")

    while True:
        query = input("\n🔍 Enter your search query: ")
        print(f"🛠️ Debug: User entered -> {query}")  # ✅ Check if input is being read

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        best_entry, best_score = find_best_match(query, embedded_data)
        print(f"🛠️ Debug: Best Match Found -> {best_entry}")  # ✅ Check if match is found

        response = generate_response(query, best_entry)
        print(f"\n💬 AI Response: {response}")  # ✅ Ensure response is printed

