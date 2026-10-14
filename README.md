# ⚡ SmartDoc RAG — AI Document Intelligence & Research Assistant

[![Machine Learning](https://img.shields.io/badge/ML-PyTorch%20%7C%20Sentence--Transformers-10b981?style=for-the-badge&logo=pytorch)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![RAG Pipeline](https://img.shields.io/badge/RAG-Qdrant%20Vector%20DB-06b6d4?style=for-the-badge&logo=qdrant)](https://qdrant.tech)
[![Generative AI](https://img.shields.io/badge/Gen%20AI-Google%20Gemini%201.5%20Flash-a855f7?style=for-the-badge&logo=google)](https://aistudio.google.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

A production-ready **Document Intelligence & Semantic Research Assistant** built from scratch to showcase the integration of **Machine Learning (ML)**, **Retrieval-Augmented Generation (RAG)**, and **Generative AI (Gen AI)**.

---

## 🎯 1. Project Purpose & Problem Statement

### The Problem:
Standard Large Language Models (ChatGPT, Gemini) generate answers based on general web pre-training data. When organizations or users need answers from **private documents** (such as internal HR policies, legal contracts, technical manuals, or research papers):
1. Public LLMs do not have access to private documents.
2. Pasting entire 50-page documents into prompts leads to context window overflow and severe **hallucinations** (making up false facts).

### The Solution:
**SmartDoc RAG** connects your custom documents directly to an in-memory vector database using local PyTorch deep learning embeddings. It performs real-time semantic similarity retrieval to fetch the exact relevant paragraphs and feeds them into **Google Gemini** to generate **100% fact-grounded answers with source citations**.

---

## 🧠 2. The 3 Technical Pillars

### 1. 🤖 Machine Learning Layer (PyTorch & Sentence-Transformers)
* **Model**: `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU.
* **Dense Vectors**: Encodes text chunks into **384-dimensional mathematical vectors**.
* **Vector Math**: Computes high-dimensional **Cosine Similarity** between query vectors and document chunk vectors.
* **Zero Cost**: Embeddings are calculated locally without paying for embedding APIs.

### 2. 📚 RAG Pipeline Layer (Qdrant Vector Database)
* **Chunking**: Splits long documents into overlapping semantic windows (e.g. 80 words with 15-word overlap) to preserve context across boundaries.
* **Vector Index**: Uses an in-memory **Qdrant** collection configured with `Distance.COSINE`.
* **Top-K Retrieval**: Extracts the Top-3 most semantically relevant text chunks based on similarity scores.

### 3. ✨ Generative AI Layer (Google Gemini)
* **Foundation LLM**: **Google Gemini 1.5 Flash** (fast inference, large context window).
* **Prompt Grounding**: Strict system prompt constraints that force the model to answer **ONLY** using retrieved context chunks.
* **Citation & Attribution**: Answers include source attribution (`[Source 1]`, `[Source 2]`) and similarity match metrics.

---

## 📐 3. End-to-End Architecture Flow

```text
 ┌────────────────┐
 │ User Document  │
 └───────┬────────┘
         ▼
 ┌────────────────┐
 │ Text Chunker   │ ➔ Splits into overlapping 80-word segments
 └───────┬────────┘
         ▼
 ┌────────────────┐
 │ ML Embedder    │ ➔ PyTorch all-MiniLM-L6-v2 (384-dimensional dense vectors)
 └───────┬────────┘
         ▼
 ┌────────────────┐
 │ Qdrant Vector  │ ➔ Stored in in-memory vector index (Cosine Distance)
 └────────────────┘
         ▲
         │ (Cosine Similarity Search)
 ┌───────┴────────┐
 │ User Question  │ ➔ Encoded into 384-dim query vector
 └────────────────┘
         │
         ▼ (Top-K Chunks Retrieved)
 ┌───────────────────────────────────────────────┐
 │ Prompt Engineering (Context + Instructions)   │
 └───────────────────────┬───────────────────────┘
                         ▼
 ┌───────────────────────────────────────────────┐
 │ Google Gemini 1.5 Flash (Grounded Generation) │
 └───────────────────────┬───────────────────────┘
                         ▼
 ┌───────────────────────────────────────────────┐
 │ Grounded Answer + ML Match % + Source Badges  │
 └───────────────────────────────────────────────┘
```

---

## 🚀 4. Quick Start Guide

### Prerequisites
* **Python 3.10+** installed on your system.
* A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Amit1557/smart_research.git
   cd smart_research
   ```

2. **Configure your API key**:
   In `backend/.env`, set your key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   ```

3. **Run using the 1-Click script (Windows)**:
   Double-click `run.bat` or run:
   ```powershell
   .\run.bat
   ```

   *Or run manually via terminal:*
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

4. **Open in Browser**:
   Navigate to **`http://localhost:8000`** in your browser.

---

## 🎤 5. Interview Pitch & Technical Q&A

### 30-Second Elevator Pitch:
> *"I built SmartDoc RAG, an intelligent document research assistant that combines Machine Learning, Vector RAG, and Generative AI. It uses a local PyTorch Sentence-Transformer to embed documents into 384-dimensional dense vectors, an in-memory Qdrant database to retrieve relevant chunks via Cosine Similarity, and Google Gemini to generate fact-grounded responses with zero hallucination and source citations."*

### Key Technical Questions:
1. **Why use local embeddings instead of OpenAI / Gemini Embeddings?**
   * *Answer*: Using local `all-MiniLM-L6-v2` runs on CPU with zero latency overhead from network calls and zero cost per API token.
2. **How does Cosine Similarity work here?**
   * *Answer*: Cosine similarity computes the dot product of two normalized vectors: $\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$. Values close to 1.0 indicate high semantic similarity regardless of document length.
3. **How does this prevent hallucination?**
   * *Answer*: We use Grounded Prompt Engineering. Gemini is explicitly instructed to answer strictly based on the retrieved context chunks and to cite each source chunk directly.

---

## 📁 Repository File Structure

```text
smart_research/
├── backend/
│   ├── main.py          # Unified FastAPI backend (ML Embeddings + Qdrant + Gemini)
│   ├── requirements.txt # Minimal, optimized Python dependencies
│   ├── .env.example     # Environment template
│   └── .env             # Your local API keys (gitignored)
├── frontend/
│   ├── index.html       # Sleek, responsive dark-mode UI
│   ├── style.css        # Glassmorphic modern CSS design system
│   └── app.js           # Client logic with sample document presets
├── .gitignore           # Ignores venvs, cache, and secrets
├── README.md            # Comprehensive documentation & architecture
└── run.bat              # 1-click startup script for Windows
```

---

## 📜 License
MIT License — Free to use for academic, portfolio, and commercial purposes.
