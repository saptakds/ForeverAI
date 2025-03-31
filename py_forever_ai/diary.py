import fitz  # PyMuPDF for PDF text extraction
import re    # Regular expressions for detecting dates
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer
import json

# Load an offline embedding model (InstructorXL)
EMBEDDING_MODEL = "hkunlp/instructor-xl"
embedder = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF file
    text = ""  
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to clean extracted text
def clean_text(text):
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace("\n", " ")  
    cleaned_text = " ".join(cleaned_text.split())  
    return cleaned_text

# Function to split text by date-based entries
def split_text_by_date(text):
    date_pattern = r"(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b)"
    entries = re.split(date_pattern, text)
    
    diary_entries = []
    for i in range(1, len(entries), 2):  
        date = entries[i].strip()
        entry_text = entries[i + 1].strip() if i + 1 < len(entries) else ""
        diary_entries.append({"date": date, "text": entry_text})

    return diary_entries

# Function to split long diary entries into smaller meaning-based chunks
def split_into_meaningful_chunks(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentences
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

# Function to generate embeddings for diary chunks
def generate_embeddings(diary_chunks):
    embedded_entries = []
    
    for entry in diary_chunks:
        text_to_embed = f"{entry['date']}: {entry['chunk_text']}"
        embedding_vector = embedder.get_text_embedding(text_to_embed)  # Convert to vector
        
        embedded_entries.append({
            "date": entry["date"],
            "chunk_text": entry["chunk_text"],
            "embedding": embedding_vector  # Convert tensor to list for storage
        })

    return embedded_entries

# ---------------- MAIN SCRIPT ---------------- #
if __name__ == "__main__":
    pdf_path = "/Users/saptakds/Documents/WIP Projects/Forever AI/Diary Content.pdf"  # Change this to your actual PDF file

    # Extract, clean, and split text
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    diary_entries = split_text_by_date(cleaned_text)

    # Apply semantic chunking to each diary entry
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
