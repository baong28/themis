import os
import sqlite3
import pickle
from typing import List, Dict
import numpy as np
import faiss
import streamlit as st
from prompts.builder import *
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# from google import genai 
# from google.genai.errors import APIError

# ======================
# ⚙️ CONFIG
# ======================
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
#GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
INDEX_PATH = "data/faiss_store/index.faiss"
META_PATH = "data/faiss_store/metadata.pkl"
SQLITE_DB_PATH = "data/faiss_store/metadata.db"

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
client = OpenAI(api_key=OPENAI_API_KEY) # api_key=GEMINI_API_KEY

# ======================
# 🔹 Load metadata from SQLite
# ======================
def load_metadata_from_sqlite() -> List[Dict]:
    """
    Load all chunk metadata from SQLite into a list, preserving order of insertion.
    """
    if not os.path.exists(SQLITE_DB_PATH):
        raise FileNotFoundError(f"❌ SQLite database not found: {SQLITE_DB_PATH}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM chunks ORDER BY rowid")
    rows = cur.fetchall()
    conn.close()

    documents = []
    for row in rows:
        documents.append({
            "page_content": row["content"],
            "metadata": {
                "bates_id": row["id"],
                "source": row["filename"],
                "path": row["path"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "chunk_chars": row["chunk_chars"],
                "has_ocr": bool(row["has_ocr"]),
                "collection_id": row["collection_id"]
            }
        })
    print(f"📚 Loaded {len(documents)} chunks from SQLite.")
    return documents

# ======================
# 🔍 QUERY SIMILAR DOCUMENTS
# ======================

# def query_similar_documents(query: str, index, embeddings_model, documents: List[Dict], top_k: int = 8):
#     """
#     Retrieve top_k most relevant document chunks from FAISS index.
#     Works with metadata.pkl structured as {"documents": [{"page_content": ..., "metadata": {...}}, ...]}
#     """

#     if not documents or index is None:
#         print("⚠️ No documents or FAISS index loaded.")
#         return []

#     # --- 1️⃣ Encode query
#     query_vector = embeddings_model.encode([query], normalize_embeddings=True)
#     query_vector = np.array(query_vector, dtype="float32")
#     if query_vector.ndim == 1:
#         query_vector = query_vector.reshape(1, -1)

#     # --- 2️⃣ FAISS search
#     distances, indices = index.search(query_vector, top_k)

#     # --- 3️⃣ Build result list
#     results = []
#     for idx, dist in zip(indices[0], distances[0]):
#         if idx == -1 or idx >= len(documents):
#             continue
#         doc = documents[idx]
#         metadata = doc.get("metadata", {})
#         content = doc.get("page_content", "")

#         results.append({
#             "content": content,
#             "metadata": {
#                 "source": metadata.get("source", "Unknown"),
#                 "path": metadata.get("path", ""),
#                 "page": metadata.get("page", None),
#                 "bates": metadata.get("bates_id", metadata.get("bates", None)), 
#                 "custodian": metadata.get("custodian", None),
#                 "collection_id": metadata.get("collection_id", None),
#                 "distance": float(dist)
#             }
#         })
#     return results

# ======================
# 💬 ASK (to OpenAI)
# ======================
# def ask(question: str):
#     """
#     Send the question to OpenAI with relevant FAISS context.
#     """

#     # --- Load FAISS index
#     if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
#         raise FileNotFoundError("❌ FAISS index or metadata missing.")

#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         data = pickle.load(f)

#     documents = data.get("documents", [])
#     if not documents:
#         print("⚠️ No documents found in metadata.pkl")
#         return "No documents available."

#     # --- Retrieve relevant context
#     context_docs = query_similar_documents(
#         query=question,
#         index=index,
#         embeddings_model=embedding_model,
#         documents=documents,
#         top_k=8
#     )

#     # Extract only text chunks for build_prompt
#     context_texts = [d["content"] for d in context_docs]

#     # --- Build structured prompt
#     prompt = build_prompt(question, context_texts)

#     # --- Query OpenAI
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are Themis – a Legal Discovery Assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return response.choices[0].message.content.strip()


# ======================
# 🔍 QUERY SIMILAR DOCUMENTS
# ======================
def query_similar_documents(query: str, index, embeddings_model, documents: List[Dict], top_k: int = 8):
    """
    Retrieve top_k most relevant document chunks from FAISS index,
    aligned with metadata loaded from SQLite.
    """
    if not documents or index is None:
        print("⚠️ No documents or FAISS index loaded.")
        return []

    # 1️⃣ Encode query
    query_vector = embeddings_model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype="float32")
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # 2️⃣ FAISS search
    distances, indices = index.search(query_vector, top_k)

    # 3️⃣ Build result list
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(documents):
            continue
        doc = documents[idx]
        metadata = doc["metadata"]
        content = doc["page_content"]

        results.append({
            "content": content,
            "metadata": {
                "bates": metadata.get("bates_id"),
                "source": metadata.get("source"),
                "path": metadata.get("path"),
                "page": metadata.get("page"),
                "distance": float(dist)
            }
        })
    return results

# ======================
# 💬 ASK (to OpenAI)
# ======================
def ask(question: str):
    """
    Send the question to OpenAI with relevant FAISS context.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(SQLITE_DB_PATH):
        raise FileNotFoundError("❌ FAISS index or SQLite metadata missing.")

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata from SQLite
    documents = load_metadata_from_sqlite()
    if not documents:
        print("⚠️ No documents found in SQLite metadata.")
        return "No documents available."

    # Retrieve relevant context
    context_docs = query_similar_documents(
        query=question,
        index=index,
        embeddings_model=embedding_model,
        documents=documents,
        top_k=8
    )

    # Build citation section
    citations = []
    for d in context_docs:
        meta = d["metadata"]
        citations.append(f"Citation: {meta['bates']} (Page {meta['page']})\nSource: {meta['source']}")

    citation_text = "\n\n".join(citations)
    context_texts = [d["content"] for d in context_docs]

    # Build prompt
    prompt = build_prompt(question, context_texts)
    prompt += "\n\n---\n" + citation_text

    # Query OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Themis – a Legal Discovery Assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()