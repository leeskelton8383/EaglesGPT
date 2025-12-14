# rag_utils.py

import sys
import os
import json
from dotenv import load_dotenv

# -----------------------------------
# Environment setup
# -----------------------------------

# Load .env file
load_dotenv()

# Force all HTTPS libraries to use corporate cert
os.environ["REQUESTS_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE")
os.environ["CURL_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE")
os.environ["SSL_CERT_FILE"] = os.getenv("REQUESTS_CA_BUNDLE")

from paths import (
    CHUNKS_JSONL_PATH,
    FAISS_INDEX_PATH,
    METADATA_JSON_PATH,
    NARRATIVE_DIR,
)

# -----------------------------------
# 1. Chunking utilities
# -----------------------------------

def chunk_text_fixed(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks.
    Each chunk has `chunk_size` words, with `overlap` words shared with the next chunk.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def load_and_chunk_wikipedia(folder_path, output_path, chunk_size=300, overlap=50):
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                doc_title = filename.replace(".txt", "")

                chunks = chunk_text_fixed(
                    text,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_title}_{str(i+1).zfill(3)}"
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "text": chunk,
                    })

    # --- Print first 3 chunks ---
    print("\n--- Example Chunks (JSON) ---")
    for chunk in all_chunks[:3]:
        print(json.dumps(chunk, indent=2))

    # --- Save all chunks to JSONL ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for chunk in all_chunks:
            out_file.write(json.dumps(chunk) + "\n")

    print(f"\n✅ Saved {len(all_chunks)} chunks to {output_path}")
    return all_chunks


# -----------------------------------
# 2. Embedding + indexing utilities
# -----------------------------------

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
import httpx


def load_chunks_from_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    print(f"🔁 Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()
    return chunks


def build_faiss_index(embedded_chunks, index_path, metadata_path):
    embeddings = np.array(
        [c["embedding"] for c in embedded_chunks]
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to {index_path}")

    metadata = [
        {k: c[k] for k in ["chunk_id", "doc_title", "text"]}
        for c in embedded_chunks
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to {metadata_path}")


# -----------------------------------
# Optional: run end-to-end when executed directly
# -----------------------------------

if __name__ == "__main__":

    # Step 1: Chunk narrative docs
    chunks = load_and_chunk_wikipedia(
        folder_path=NARRATIVE_DIR,
        output_path=CHUNKS_JSONL_PATH,
    )

    # Step 2: Load chunks
    chunks = load_chunks_from_jsonl(CHUNKS_JSONL_PATH)
    print(f"📦 Loaded {len(chunks)} chunks.")

    # Step 3: Embed chunks
    embedded_chunks = embed_chunks(chunks)

    # Step 4: Build FAISS index
    build_faiss_index(
        embedded_chunks,
        FAISS_INDEX_PATH,
        METADATA_JSON_PATH,
    )
