"""
SmartDoc RAG - Intelligent Document Research Assistant
======================================================
Showcases 3 Core Pillars:
1. Machine Learning (ML)   : PyTorch Sentence-Transformers (all-MiniLM-L6-v2) for 384-dim dense embeddings & Cosine Similarity.
2. RAG Pipeline            : Document Chunking + Qdrant Vector Database for semantic search.
3. Generative AI (Gen AI)  : Google Gemini API (gemini-1.5-flash) with grounded prompt engineering.
"""

import os
import time
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Qdrant Vector Database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Sentence Transformers (PyTorch-based ML model)
from sentence_transformers import SentenceTransformer

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-rag")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MACHINE LEARNING & EMBEDDINGS LAYER
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

logger.info(f"Loading ML Embedding Model: {EMBEDDING_MODEL_NAME}...")
# Loads local transformer model onto CPU - zero API cost
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info("ML Model loaded successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR DATABASE (RAG LAYER - Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "smart_doc_knowledge"

# Use in-memory Qdrant instance (Zero configuration / no Docker required)
qdrant = QdrantClient(":memory:")

def init_vector_db():
    """Initializes or resets the Qdrant collection."""
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Qdrant collection initialized.")

init_vector_db()

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATIVE AI (LLM LAYER - Google Gemini)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

def call_gemini(prompt: str) -> str:
    """Invokes Google Gemini with the grounded RAG context."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        return (
            "⚠️ **Gemini API Key is missing!**\n\n"
            "To get live Gen AI answers, create a free API key at "
            "[Google AI Studio](https://aistudio.google.com/app/apikey) "
            "and add it to `backend/.env`.\n\n"
            "*(Notice: The ML Embeddings and RAG Vector Retrieval above are working 100% locally!)*"
        )

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or "No response generated."
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return f"Error contacting Gemini API: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER HELPER (RAG)
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splits a document into overlapping semantic chunks."""
    text = text.strip()
    if not text:
        return []
    
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words) - overlap:
            break
    return chunks if chunks else [text]

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartDoc RAG API",
    description="Demonstration of Machine Learning, RAG, and Generative AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class IndexRequest(BaseModel):
    document_title: str = Field(default="Uploaded Document", description="Document Title")
    text: str = Field(..., description="Full text content of the document to index")
    chunk_size: int = Field(default=80, description="Words per chunk")

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question about the indexed document")
    top_k: int = Field(default=3, description="Number of context chunks to retrieve")

# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "ml_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "vector_db": "Qdrant (In-Memory)",
        "genai_model": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_")),
    }

@app.post("/api/index")
def index_document(req: IndexRequest):
    """
    RAG Step 1 & 2:
    - Splits text into chunks.
    - Generates 384-dimensional dense vectors using SentenceTransformer.
    - Upserts vectors into Qdrant.
    """
    start_time = time.time()
    chunks = chunk_text(req.text, chunk_size=req.chunk_size, overlap=15)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    # 1. ML Embedding generation
    vectors = embed_model.encode(chunks, normalize_embeddings=True).tolist()

    # 2. Vector DB upsert
    points = [
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "document_title": req.document_title,
                "chunk_id": idx,
                "text": chunk,
            },
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "message": f"Successfully indexed '{req.document_title}'",
        "chunks_indexed": len(chunks),
        "embedding_dimensions": EMBEDDING_DIM,
        "ml_model_used": EMBEDDING_MODEL_NAME,
        "time_taken_ms": elapsed_ms,
    }

@app.post("/api/query")
def query_rag(req: QueryRequest):
    """
    RAG Step 3 & 4 + Gen AI:
    - Encodes user question into vector.
    - Performs Cosine Similarity search in Qdrant for Top-K chunks.
    - Constructs grounded prompt.
    - Calls Google Gemini to generate fact-based response with citations.
    """
    start_time = time.time()

    # Check if knowledge base has data
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty! Please index a document first.",
        )

    # 1. ML Step: Encode question
    query_vector = embed_model.encode(req.question, normalize_embeddings=True).tolist()

    # 2. RAG Retrieval Step: Cosine Similarity search
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=req.top_k,
    )

    retrieved_sources = []
    context_blocks = []
    similarity_scores = []

    for rank, hit in enumerate(search_results, 1):
        score_percent = round(hit.score * 100, 1)
        similarity_scores.append(score_percent)
        chunk_text_content = hit.payload.get("text", "")
        doc_name = hit.payload.get("document_title", "Document")

        retrieved_sources.append({
            "rank": rank,
            "document": doc_name,
            "chunk_id": hit.payload.get("chunk_id", 0),
            "similarity_score": score_percent,
            "text": chunk_text_content,
        })

        context_blocks.append(f"[Source {rank} - {doc_name} (Match: {score_percent}%)]:\n{chunk_text_content}")

    combined_context = "\n\n".join(context_blocks)

    # 3. Gen AI Step: Prompt Engineering with Grounded Context
    prompt = f"""You are an intelligent, fact-grounded research assistant.
You MUST answer the question using ONLY the provided reference sources below.
If the answer cannot be found in the reference sources, clearly state: "The provided document does not contain this information." Do NOT hallucinate.

---
REFERENCE SOURCES (Retrieved via Semantic Vector Search):
{combined_context}
---

USER QUESTION:
{req.question}

INSTRUCTIONS:
1. Provide a direct, concise, and professional answer.
2. Cite the exact Source numbers [Source 1], [Source 2] for every key fact.
3. Keep the tone factual and authoritative.
"""

    llm_answer = call_gemini(prompt)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "answer": llm_answer,
        "ml_metrics": {
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIM,
            "highest_similarity_score": max(similarity_scores) if similarity_scores else 0,
            "latency_ms": elapsed_ms,
        },
        "retrieved_sources": retrieved_sources,
    }

@app.post("/api/reset")
def reset_database():
    """Resets the vector database memory."""
    init_vector_db()
    return {"message": "Vector database reset successfully."}

# Mount frontend directory for easy serving
FRONTEND_PATH = BASE_DIR.parent / "frontend"
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(FRONTEND_PATH / "index.html")
