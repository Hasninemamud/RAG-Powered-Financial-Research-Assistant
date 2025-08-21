Perfect timing 🚀 — let’s draft a **clear, professional README-style documentation** for your project.
This will explain **what it is, how it works, how to run it, and how to use it**.

---

# 📘 JVAI Policy Chatbot — Documentation

## 📖 Overview

The **JVAI Policy Chatbot** is an intelligent document assistant that allows users to:

* Upload **PDF policy documents**
* Ask **questions in natural language**
* Get answers from either:

  * **Extractive mode** → retrieves exact snippets + page numbers
  * **LLM mode** → summarizes context using an external LLM (Hyperbolic API)

It combines:

* **FastAPI** → backend REST API + HTML endpoints
* **FAISS** → semantic vector search for document retrieval
* **Hyperbolic LLM API** → natural language summarization
* **Streamlit** → user-friendly web UI for uploads + Q\&A

---

## 🏗️ Project Structure

```
jvai-policy-chatbot/
│
├── .venv/                  # Virtual environment (not in git)
├── requirements.txt         # Dependencies
├── .env                     # Environment variables (API keys, config)
├── README.md                # Documentation (this file)
│
├── data/                    # Uploaded PDFs + FAISS indexes
│   ├── policy.pdf
│   ├── index.faiss
│   └── store.pkl
│
├── templates/               # HTML templates (FastAPI demo UI)
│   └── index.html
│
├── ui.py                    # Streamlit app (frontend for users)
│
└── src/                     # Backend source code
    ├── __init__.py
    ├── app.py               # FastAPI app (endpoints, routes)
    ├── config.py            # Config/env management
    ├── ingest.py            # PDF → text → embeddings → FAISS
    ├── retriever.py         # Semantic search with FAISS
    ├── memory.py            # Conversation memory manager
    ├── llm.py               # LLM integration (Hyperbolic API)
    └── chatbot.py           # CLI chatbot (optional)
```

---

## ⚙️ Installation

### 1. Clone project

```bash
git clone https://github.com/Hasninemamud/Policy-Chatbot
cd jvai-policy-chatbot
```

### 2. Setup virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file:

```
HYPERBOLIC_API_KEY=your-secret-key
DATA_DIR=data
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
MODEL_NAME=all-MiniLM-L6-v2
```

---

## 🚀 Running the Project

### Option A — FastAPI Backend

Start server:

```bash
uvicorn src.app:app --reload --port 8000
```

* API docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Simple HTML form → [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Option B — Streamlit UI

Run Streamlit:

```bash
streamlit run ui.py
```

* Open → [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 🖥️ Usage Guide

### 1. Upload PDF

* Upload a **policy document** (e.g., `policy.pdf`)
* The system extracts text, chunks it, embeds it, and stores it in FAISS

### 2. Ask a Question

* Type any question in natural language, e.g.:

  ```
  What is the budget allocation for infrastructure?
  ```
* Choose:

  * **Extractive Answer** → shows top matching snippets + page numbers
  * **LLM Answer** → sends context + question to Hyperbolic LLM API for summarized answer

### 3. Get Results

* Answer (extractive or LLM summary)
* Source citations (page + similarity score)

---

## 🧩 API Endpoints

### `POST /upload_pdf`

Upload and process a PDF.

```bash
curl -X POST "http://127.0.0.1:8000/upload_pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/policy.pdf"
```

### `POST /ask`

Ask a question.

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"session_id":"demo","question":"What is the budget allocation for infrastructure?","use_llm":true}'
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest -q
```

---

## 📌 Future Improvements

* Chat-style UI (multi-turn conversation in Streamlit)
* Support for multiple file formats (DOCX, TXT)
* Multi-document retrieval
* Role-based access with authentication



Do you want me to also generate the **`requirements.txt`** file for you (with exact dependencies) so your documentation + project are complete?
