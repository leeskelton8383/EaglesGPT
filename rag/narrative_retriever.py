# narrative_retriever.py

import os
import re
import json
import faiss
import httpx
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from paths import FAISS_INDEX_PATH, METADATA_JSON_PATH

# -----------------------------------
# 🔑 Load API Key Securely
# -----------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY environment variable not set.")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(verify=os.environ.get("REQUESTS_CA_BUNDLE"))
)

# -----------------------------------
# 🔍 Retrieve Chunks
# -----------------------------------

def extract_years(text):
    return re.findall(r'20\d{2}', text)


def retrieve_narrative_chunks(
    question,
    index_path,
    metadata_path,
    model_name='all-MiniLM-L6-v2',
    top_k=5
):
    print(f"\n🔍 Retrieving top {top_k} chunks for question: {question}")

    target_years = set(extract_years(question))

    index = faiss.read_index(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    model = SentenceTransformer(model_name)
    query_vector = model.encode([question]).astype('float32')

    distances, indices = index.search(query_vector, top_k * 3)
    candidates = [metadata[i] for i in indices[0]]

    def score(chunk):
        title = chunk['doc_title']
        return 0 if any(y in title for y in target_years) else 1

    candidates.sort(key=score)
    results = candidates[:top_k]

    print(
        f"\n📄 Retrieved Chunks "
        f"(boosted by years: {', '.join(target_years) if target_years else 'None'}):\n"
    )

    for i, r in enumerate(results):
        print(f"[{i+1}] ({r['chunk_id']}) from {r['doc_title']}")
        print(r['text'][:300] + "\n---\n")

    return results


# -----------------------------------
# 🔁 Public Wrapper (for router / pipeline)
# -----------------------------------

def get_relevant_narrative_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Wrapper to retrieve narrative chunks given a user question.

    Args:
        question (str): The natural language question.
        top_k (int): Number of top chunks to return.

    Returns:
        List of dicts: Each chunk has 'chunk_id', 'doc_title', 'text'
    """
    return retrieve_narrative_chunks(
        question=question,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_JSON_PATH,
        top_k=top_k
    )


# -----------------------------------
# OPTIONAL: Manual test run
# -----------------------------------

if __name__ == "__main__":

    question = "Which was more disappointing, the 2020 or 2023 Eagles season?"

    chunks = get_relevant_narrative_chunks(question, top_k=5)

    print("\n✅ Retrieved", len(chunks), "chunks")


# %%
#get_relevant_narrative_chunks("Which was more disappointing, the 2020 or 2023 Eagles season?", top_k=5)



