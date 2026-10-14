"""
SmartDoc RAG - Intelligent Document Research Assistant (v2.1)
============================================================
Features:
1. Multi-Document Knowledge Base (Upload multiple PDFs, DOCXs, TXTs, MDs; query across all files).
2. 1-Click Executive Summary & Key Takeaways Generator (Gemini + RAG).
3. Machine Learning (PyTorch HuggingFace Transformers all-MiniLM-L6-v2 for 384-d dense embeddings).
4. Vector Database (Qdrant in-memory with Cosine Similarity and metadata filtering).
5. Generative AI (Google Gemini 2.5 Flash / 1.5 Flash with source-grounded citations).
"""

import os
import io
import time
import uuid
import re
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv, set_key

# Qdrant Vector Database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Pure PyTorch & Transformers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Document parsers
import pypdf
import docx

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-rag")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MACHINE LEARNING & EMBEDDINGS LAYER (PyTorch + Transformers)
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class PyTorchTransformerEmbedder:
    """Native PyTorch transformer embedder (runs on CPU with normalized vectors)."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading PyTorch Transformer ML Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        logger.info("ML Model loaded successfully.")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

embed_model = PyTorchTransformerEmbedder()

# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR DATABASE (RAG LAYER - Qdrant)
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "smart_doc_knowledge"
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

def call_gemini(prompt: str, override_key: Optional[str] = None, override_model: Optional[str] = None) -> str:
    """Invokes Google Gemini with the grounded RAG context."""
    api_key = (override_key or GEMINI_API_KEY).strip()
    model_name = (override_model or GEMINI_MODEL).strip() or "gemini-2.5-flash"

    if not api_key or api_key.startswith("your_"):
        return (
            "⚠️ **Gemini API Key is missing!**\n\n"
            "Please open **Settings (⚙️)** in the top bar and paste your Google Gemini API key to enable live AI responses.\n\n"
            "*(Your local PyTorch ML embeddings and Qdrant semantic vector search are working 100% locally!)*"
        )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text or "No response generated by the model."
    except Exception as e:
        logger.error(f"Gemini API Error ({model_name}): {e}")
        return f"⚠️ **Gemini API Error ({model_name})**: {str(e)}\n\nPlease check your API key in Settings."

# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-DOCUMENT LIBRARY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
# Dictionary mapping doc_id -> metadata
documents_library: Dict[str, Dict[str, Any]] = {}

def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Extracts text page-by-page from PDF."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        clean_text = page_text.strip()
        if clean_text:
            pages_text.append(f"[Page {idx + 1}]\n{clean_text}")
    
    total_pages = len(reader.pages)
    combined = "\n\n".join(pages_text) if pages_text else ""
    return combined, total_pages

def extract_text_from_docx(file_bytes: bytes) -> tuple[str, int]:
    """Extracts text from DOCX."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs), max(1, len(paragraphs) // 10)

def extract_text_from_raw(file_bytes: bytes) -> tuple[str, int]:
    """Extracts plain text with multiple encodings."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            text = file_bytes.decode("utf-8", errors="replace")
    return text.strip(), 1

def chunk_text_with_metadata(text: str, chunk_size: int = 100, overlap: int = 20) -> List[Dict[str, Any]]:
    """Splits document text into overlapping chunks, tracking detected page numbers."""
    text = text.strip()
    if not text:
        return []

    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text_str = " ".join(chunk_words)

        # Detect page number tag if present
        page_match = re.search(r"\[Page\s+(\d+)\]", chunk_text_str)
        page_num = int(page_match.group(1)) if page_match else None

        # Clean out internal page markers for cleaner embeddings
        cleaned_chunk = re.sub(r"\[Page\s+\d+\]\n?", "", chunk_text_str).strip()

        if cleaned_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": cleaned_chunk,
            })
            chunk_id += 1

        i += max(1, chunk_size - overlap)
        if i >= len(words):
            break

    return chunks

def generate_suggested_questions(text: str, filename: str) -> List[str]:
    """Generates 3-4 relevant quick questions based on document text."""
    lower = text.lower()
    if "leave" in lower or "policy" in lower or "employee" in lower:
        return [
            "What is the leave, vacation, and work from home policy?",
            "What health insurance and employee perks are provided?",
            "What is the notice period and resignation protocol?",
        ]
    elif "revenue" in lower or "financial" in lower or "quarter" in lower or "market" in lower:
        return [
            "What are the key financial metrics and revenue figures?",
            "What were the major growth drivers or unexpected costs?",
            "What are the future projections and strategic targets?",
        ]
    elif "research" in lower or "abstract" in lower or "transformer" in lower or "algorithm" in lower:
        return [
            "What is the core methodology and problem statement?",
            "What were the experimental findings and accuracy metrics?",
            "How does this approach compare with baseline techniques?",
        ]
    return [
        f"What is the main summary and core objective of {filename}?",
        "What are the key points and highlights mentioned?",
        "What important requirements, guidelines, or deadlines are specified?",
    ]

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartDoc RAG API",
    description="Multi-Document Intelligence & Research Platform (PyTorch + Qdrant + Gemini)",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class IndexRequest(BaseModel):
    document_title: str = Field(default="Pasted Document")
    text: str = Field(...)
    chunk_size: int = Field(default=90)
    chunk_overlap: int = Field(default=20)

class QueryRequest(BaseModel):
    question: str = Field(...)
    top_k: int = Field(default=4)
    doc_id: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

class SummarizeRequest(BaseModel):
    doc_id: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

class SettingsUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "status": "healthy",
        "ml_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "vector_db": "Qdrant (In-Memory)",
        "genai_model": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_")),
        "total_documents": len(documents_library),
        "total_points_indexed": collection_info.points_count if collection_info else 0,
    }

@app.get("/api/documents")
def get_documents():
    """Returns list of all indexed documents in the multi-document library."""
    return list(documents_library.values())

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a specific document from Qdrant vector memory and library."""
    if doc_id not in documents_library:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc_info = documents_library[doc_id]
    
    # Delete points in Qdrant matching doc_id
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
    except Exception as e:
        logger.warning(f"Error deleting points for {doc_id}: {e}")

    del documents_library[doc_id]
    return {
        "success": True,
        "message": f"Deleted '{doc_info['filename']}' from vector memory.",
        "remaining_documents": len(documents_library),
    }

@app.get("/api/settings")
def get_settings():
    """Returns current system configuration and available models."""
    has_key = bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"))
    masked_key = ""
    if has_key:
        if len(GEMINI_API_KEY) > 8:
            masked_key = f"{GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}"
        else:
            masked_key = "••••••••"

    collection_info = qdrant.get_collection(COLLECTION_NAME)

    return {
        "api_key_configured": has_key,
        "api_key_masked": masked_key,
        "current_model": GEMINI_MODEL,
        "available_models": AVAILABLE_MODELS,
        "ml_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "points_indexed": collection_info.points_count if collection_info else 0,
        "total_documents": len(documents_library),
    }

@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest):
    """Updates API key and model dynamically without server restart."""
    global GEMINI_API_KEY, GEMINI_MODEL
    updated = {}

    if req.gemini_api_key is not None:
        key_val = req.gemini_api_key.strip()
        GEMINI_API_KEY = key_val
        os.environ["GEMINI_API_KEY"] = key_val
        try:
            set_key(str(ENV_PATH), "GEMINI_API_KEY", key_val)
        except Exception as e:
            logger.warning(f"Could not write key to .env: {e}")
        updated["gemini_api_key"] = "Updated successfully"

    if req.gemini_model is not None:
        model_val = req.gemini_model.strip()
        if model_val:
            GEMINI_MODEL = model_val
            os.environ["GEMINI_MODEL"] = model_val
            try:
                set_key(str(ENV_PATH), "GEMINI_MODEL", model_val)
            except Exception as e:
                logger.warning(f"Could not write model to .env: {e}")
            updated["gemini_model"] = model_val

    return {"message": "Settings updated successfully", "changes": updated}

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(100),
    chunk_overlap: int = Form(20),
):
    """
    Direct File Upload from Laptop (PDF, DOCX, TXT, MD, CSV, etc.):
    - Supports Multi-Document additions without erasing existing files!
    """
    start_time = time.time()
    filename = file.filename or "uploaded_document"
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ext = Path(filename).suffix.lower()
    page_count = 1
    extracted_text = ""

    try:
        if ext == ".pdf":
            extracted_text, page_count = extract_text_from_pdf(file_bytes)
        elif ext in [".docx", ".doc"]:
            extracted_text, page_count = extract_text_from_docx(file_bytes)
        else:
            extracted_text, page_count = extract_text_from_raw(file_bytes)
    except Exception as e:
        logger.error(f"Error parsing file {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file '{filename}': {str(e)}")

    if not extracted_text or not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"No readable text could be extracted from '{filename}'. (If it is a scanned image PDF, please use a PDF with selectable text).",
        )

    # Generate unique doc_id
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Chunk text
    chunks = chunk_text_with_metadata(extracted_text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document could not be chunked.")

    # Generate PyTorch dense embeddings
    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(chunk_texts)

    # Upsert into Qdrant using unique UUID point IDs
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "document_title": filename,
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "text": c["text"],
            },
        )
        for c, vector in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    words = len(extracted_text.split())
    suggested = generate_suggested_questions(extracted_text, filename)

    doc_info = {
        "doc_id": doc_id,
        "filename": filename,
        "filesize_formatted": format_bytes(len(file_bytes)),
        "filetype": ext.replace(".", "").upper() or "TXT",
        "page_count": page_count,
        "chunks_count": len(chunks),
        "char_count": len(extracted_text),
        "word_count": words,
        "indexed_at": time.strftime("%I:%M %p"),
        "suggested_questions": suggested,
        "preview": extracted_text[:400] + ("..." if len(extracted_text) > 400 else ""),
    }

    documents_library[doc_id] = doc_info

    return {
        "success": True,
        "message": f"Successfully parsed & indexed '{filename}'",
        "document": doc_info,
        "total_documents": len(documents_library),
        "time_taken_ms": elapsed_ms,
    }

@app.post("/api/index")
def index_document(req: IndexRequest):
    """Indexes raw pasted text into the multi-document library."""
    start_time = time.time()
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    chunks = chunk_text_with_metadata(raw_text, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document could not be chunked.")

    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(chunk_texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "document_title": req.document_title,
                "chunk_id": c["chunk_id"],
                "page": c["page"] or 1,
                "text": c["text"],
            },
        )
        for c, vector in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    suggested = generate_suggested_questions(raw_text, req.document_title)

    doc_info = {
        "doc_id": doc_id,
        "filename": req.document_title,
        "filesize_formatted": format_bytes(len(raw_text.encode("utf-8"))),
        "filetype": "TEXT",
        "page_count": 1,
        "chunks_count": len(chunks),
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
        "indexed_at": time.strftime("%I:%M %p"),
        "suggested_questions": suggested,
        "preview": raw_text[:400] + ("..." if len(raw_text) > 400 else ""),
    }

    documents_library[doc_id] = doc_info

    return {
        "success": True,
        "message": f"Successfully indexed '{req.document_title}'",
        "document": doc_info,
        "total_documents": len(documents_library),
        "time_taken_ms": elapsed_ms,
    }

@app.post("/api/summarize")
def summarize_document(req: SummarizeRequest):
    """
    ⚡ 1-Click Executive Summary & Key Takeaways Generator:
    - Analyzes representative chunks from the document.
    - Generates structured overview, key takeaways, figures/dates, and action items.
    """
    start_time = time.time()

    if not documents_library:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a document first.")

    target_docs = []
    if req.doc_id and req.doc_id in documents_library:
        target_docs = [documents_library[req.doc_id]]
    else:
        target_docs = list(documents_library.values())

    doc_titles = ", ".join([d["filename"] for d in target_docs])

    # Retrieve representative chunks from Qdrant
    # We query with a broad semantic vector (e.g. "overview summary key findings takeaways")
    summary_query = "executive summary core objective main findings key takeaways policies figures conclusions"
    query_vector = embed_model.encode([summary_query])[0]

    filter_query = None
    if req.doc_id:
        filter_query = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=req.doc_id))]
        )

    search_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=filter_query,
        limit=min(8, 4 * len(target_docs)),
    )
    search_results = search_response.points

    context_blocks = []
    for hit in search_results:
        doc_name = hit.payload.get("document_title", "Document")
        page_val = hit.payload.get("page")
        page_str = f"Page {page_val}" if page_val else f"Chunk #{hit.payload.get('chunk_id', 1)}"
        context_blocks.append(f"[{doc_name} ({page_str})]:\n{hit.payload.get('text', '')}")

    combined_context = "\n\n".join(context_blocks)

    prompt = f"""You are SmartDoc, an elite executive research analyst.
Generate a comprehensive, high-impact **Executive Summary & Key Takeaways** for: **{doc_titles}**.

Format your response cleanly in GitHub Markdown using these exact sections:

# ⚡ Executive Summary: {doc_titles}

### 📌 1. Overview & Core Objective
Provide a concise 2-3 paragraph synthesis explaining the fundamental purpose, background, and scope of this document.

### 🎯 2. Key Takeaways & Highlights
Provide 5 to 7 high-impact, actionable bullet points summarizing the most critical policies, findings, or directives. (Cite source pages e.g. [DocName, Page X] where applicable).

### 📊 3. Key Figures, Metrics & Deadlines
List any specific numbers, percentages, dates, dollar amounts, or quantifiable requirements mentioned in the text.

### ⚠️ 4. Important Conditions & Action Items
Highlight any obligations, restrictions, compliance rules, or next steps specified for the reader.

---
DOCUMENT REFERENCE CONTEXT:
{combined_context}
---

Write authoritative, polished, and structured Markdown."""

    summary_answer = call_gemini(
        prompt=prompt,
        override_key=req.api_key,
        override_model=req.model,
    )
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "success": True,
        "document_titles": doc_titles,
        "summary": summary_answer,
        "latency_ms": elapsed_ms,
        "chunks_analyzed": len(search_results),
    }

@app.post("/api/query")
def query_rag(req: QueryRequest):
    """
    Multi-Document RAG Semantic Retrieval & Grounded Q&A:
    1. Encodes query into 384-d vector via PyTorch.
    2. Searches Qdrant (optionally filtering by specific doc_id).
    3. Feeds cross-document context into Gemini LLM.
    4. Returns grounded answer with exact multi-document source citations.
    """
    start_time = time.time()

    if not documents_library:
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed yet! Please upload a file from your laptop first.",
        )

    # 1. PyTorch encode query
    query_vector = embed_model.encode([req.question])[0]

    # Optional filter if specific doc_id was requested
    filter_condition = None
    if req.doc_id:
        filter_condition = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=req.doc_id))]
        )

    # 2. Qdrant Cosine Similarity search
    search_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=filter_condition,
        limit=max(1, min(req.top_k, 10)),
    )
    search_results = search_response.points

    retrieved_sources = []
    context_blocks = []
    similarity_scores = []

    for rank, hit in enumerate(search_results, 1):
        score_percent = round(hit.score * 100, 1)
        similarity_scores.append(score_percent)
        chunk_text_content = hit.payload.get("text", "")
        doc_name = hit.payload.get("document_title", "Document")
        page_val = hit.payload.get("page")
        page_str = f"Page {page_val}" if page_val else f"Chunk #{hit.payload.get('chunk_id', rank)}"

        retrieved_sources.append({
            "rank": rank,
            "document": doc_name,
            "page": page_val,
            "chunk_id": hit.payload.get("chunk_id", rank),
            "similarity_score": score_percent,
            "text": chunk_text_content,
        })

        context_blocks.append(
            f"[Source {rank} | {doc_name} ({page_str}) - Relevance: {score_percent}%]:\n{chunk_text_content}"
        )

    combined_context = "\n\n".join(context_blocks)

    # 3. Grounded Prompt Engineering for Gemini
    prompt = f"""You are SmartDoc, an intelligent document research assistant.
Answer the user's question clearly, professionally, and accurately using ONLY the provided reference sources below.

CRITICAL RULES:
1. Ground your answer strictly on the provided sources. Do NOT make up facts.
2. If the document does not contain enough information to answer, state: "The uploaded document(s) do not contain this information."
3. Format your response cleanly using GitHub-flavored Markdown (bold headings, clear bullet points, code blocks).
4. Cite sources using [Source 1], [Source 2], etc., mentioning the specific document name when helpful.

---
REFERENCE SOURCES (Retrieved across documents via Qdrant Vector Search):
{combined_context}
---

USER QUESTION:
{req.question}

ANSWER:"""

    llm_answer = call_gemini(
        prompt=prompt,
        override_key=req.api_key,
        override_model=req.model,
    )
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "answer": llm_answer,
        "question": req.question,
        "ml_metrics": {
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIM,
            "highest_similarity_score": max(similarity_scores) if similarity_scores else 0,
            "latency_ms": elapsed_ms,
            "chunks_retrieved": len(retrieved_sources),
            "total_documents": len(documents_library),
        },
        "retrieved_sources": retrieved_sources,
    }

@app.post("/api/reset")
def reset_database():
    """Resets the vector database and clears all indexed documents."""
    init_vector_db()
    documents_library.clear()
    return {"message": "Knowledge base reset successfully.", "total_documents": 0}

# Mount frontend directory for easy serving
FRONTEND_PATH = BASE_DIR.parent / "frontend"
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(FRONTEND_PATH / "index.html")
