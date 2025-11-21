import os
import sqlite3
import pickle
from typing import List, Dict
import numpy as np
import faiss
import json
import streamlit as st
from prompts.builder import *
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# from google import genai 
# from google.genai.errors import APIError
from google.cloud import storage
import faiss
import tempfile

# ====================== CONFIG ======================
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
#GEMINI_API_KEY = st.secrets["gemini"]["api_key"]

GCS_BUCKET_NAME = "themis-kd-1"
GCS_METADATA_DB = "metadata.db"
GCS_FAISS_FILE = "index.faiss"
GCS_SERVICE_ACCOUNT = "data/service-account.json"

# ====================== MODEL ======================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu", trust_remote_code=True)

embedding_model = load_embedding_model()
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================
# 🔹 Load metadata from SQLite in GCS
# ======================
@st.cache_resource(show_spinner="Loading metadata from GCS…")
def load_metadata_from_gcs() -> List[Dict]:
    """
    Download metadata.db from GCS, read SQLite, convert to documents list.
    """
    # Initialize GCS client
    client = storage.Client.from_service_account_json(GCS_SERVICE_ACCOUNT)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_METADATA_DB)

    # Windows-safe temp file
    fd, tmp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Download SQLite DB
    blob.download_to_filename(tmp_db_path)

    # Read SQLite
    conn = sqlite3.connect(tmp_db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM chunks ORDER BY rowid")
    rows = cur.fetchall()
    conn.close()

    # Remove temp file
    os.remove(tmp_db_path)

    # Convert rows → documents list
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
                "collection_id": row["collection_id"],
            }
        })

    print(f"📚 Loaded {len(documents)} chunks from GCS SQLite.")
    return documents

@st.cache_resource(show_spinner="Initializing metadata…")
def cached_load_metadata():
    return load_metadata_from_gcs()

# ======================
# 🔹 Load FAISS index from GCS
# ======================
@st.cache_resource(show_spinner="Loading FAISS index from GCS…")
def load_faiss_from_gcs(bucket_name: str, file_name: str) -> faiss.Index:
    """
    Download FAISS index from GCS and load into memory.
    Windows-safe using tempfile.
    """
    # Initialize GCS client
    client = storage.Client.from_service_account_json(GCS_SERVICE_ACCOUNT)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Windows-safe temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)

    # Download FAISS index
    blob.download_to_filename(tmp_path)

    # Load FAISS index
    faiss_index = faiss.read_index(tmp_path)

    # Remove temp file
    os.remove(tmp_path)

    print(f"✨ FAISS index loaded from GCS: {bucket_name}/{file_name}")
    return faiss_index

@st.cache_resource(show_spinner="Initializing FAISS…")
def cached_load_faiss() -> faiss.Index:
    return load_faiss_from_gcs(
        bucket_name=GCS_BUCKET_NAME,
        file_name=GCS_FAISS_FILE
    )

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
# def ask(question: str):
#     """
#     Send the question to OpenAI with relevant FAISS context.
#     """
#     if not os.path.exists(INDEX_PATH) or not os.path.exists(SQLITE_DB_PATH):
#         raise FileNotFoundError("❌ FAISS index or SQLite metadata missing.")

#     # Load FAISS index
#     index = faiss.read_index(INDEX_PATH)

#     # Load metadata from SQLite
#     documents = load_metadata_from_sqlite()
#     if not documents:
#         print("⚠️ No documents found in SQLite metadata.")
#         return "No documents available."

#     # Retrieve relevant context
#     context_docs = query_similar_documents(
#         query=question,
#         index=index,
#         embeddings_model=embedding_model,
#         documents=documents,
#         top_k=8
#     )

#     # Build citation section
#     citations = []
#     for d in context_docs:
#         meta = d["metadata"]
#         citations.append(f"Citation: {meta['bates']} (Page {meta['page']})\nSource: {meta['source']}")

#     citation_text = "\n\n".join(citations)
#     context_texts = [d["content"] for d in context_docs]

#     # Build prompt
#     prompt = build_prompt(question, context_texts)
#     prompt += "\n\n---\n" + citation_text

#     # Query OpenAI
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are Themis – a Legal Discovery Assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return response.choices[0].message.content.strip()

# ====================== ASK ======================
def ask(question):
    # Load metadata
    documents = cached_load_metadata()

    # Load FAISS index từ GCS 
    faiss_index = cached_load_faiss()

    # Retrieve context from FAISS
    context_docs = query_similar_documents(
        query=question,
        index=faiss_index,
        embeddings_model=embedding_model,
        documents=documents,
        top_k=8
    )

    citations = [f"Citation: {d['metadata']['bates']} (Page {d['metadata']['page']})\nSource: {d['metadata']['source']}" for d in context_docs]
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

