"""
FastAPI app for JVAI Policy Chatbot.
Applies SOLID principles for clean design.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import shutil

# Domain services
from .config import settings
from .ingest import read_pdf_with_pages, build_corpus, embed_corpus, save_index_and_store
from .retriever import Retriever
from .memory import ConversationMemory
from .llm import call_llm




class AnswerGenerator:
    """Abstract answer generator interface."""
    def generate(self, question: str, results: list[dict]) -> str:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    """Returns raw snippets with page numbers (default)."""
    def generate(self, question: str, results: list[dict]) -> str:
        if not results:
            return "No relevant information found in the document."
        lines = [f"Q: {question}", "", "Top matches:"]
        for r in results:
            snippet = r["text"].strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            lines.append(f"- (page {r['page']}, score {r['score']:.3f}) {snippet}")
        return "\n".join(lines)


class LLMAnswerGenerator(AnswerGenerator):
    """Uses an LLM to summarize retrieved results into a natural answer."""
    def generate(self, question: str, results: list[dict]) -> str:
        return call_llm(question, results)



# Application setup


app = FastAPI(title="JVAI Policy Chatbot")
templates = Jinja2Templates(directory="templates")

# Conversation memory (per-session)
memory = ConversationMemory()
retriever: Retriever | None = None



# Request models (SRP: each endpoint has a clear schema)


class IngestReq(BaseModel):
    pdf_path: str


class AskReq(BaseModel):
    session_id: str
    question: str
    top_k: int | None = None
    use_llm: bool = False



# Endpoints


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, save it to data/, and build the FAISS index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = Path(settings.DATA_DIR) / file.filename
    save_path.parent.mkdir(exist_ok=True, parents=True)

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run ingestion pipeline
    pages = read_pdf_with_pages(save_path)
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    corpus = build_corpus(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embeddings = embed_corpus(corpus, settings.MODEL_NAME)
    save_index_and_store(embeddings, corpus, settings.index_path, settings.store_path)

    global retriever
    retriever = Retriever()

    return {"message": "PDF uploaded and ingested successfully!", "chunks": len(corpus)}


@app.post("/ingest")
def ingest(req: IngestReq):
    """
    Ingest a PDF from a given path (server-side).
    """
    pdf = Path(req.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf}")

    pages = read_pdf_with_pages(pdf)
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text in the PDF.")

    corpus = build_corpus(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embeddings = embed_corpus(corpus, settings.MODEL_NAME)
    save_index_and_store(embeddings, corpus, settings.index_path, settings.store_path)

    global retriever
    retriever = Retriever()

    return {"chunks": len(corpus), "index_path": str(settings.index_path), "store_path": str(settings.store_path)}


@app.post("/ask")
def ask(req: AskReq):
    """
    Ask a question via API.
    Supports both extractive (default) and LLM summarization.
    """
    global retriever
    if retriever is None:
        raise HTTPException(status_code=400, detail="Index not found. Please upload or ingest a PDF first.")

    # Contextualize question with memory
    q_ctx = memory.contextualize(req.session_id, req.question)
    results = retriever.search(q_ctx, top_k=req.top_k or settings.TOP_K)
    topic = memory.infer_topic(req.question)
    memory.update(req.session_id, req.question, topic)

    # Choose answer generator
    generator: AnswerGenerator = LLMAnswerGenerator() if req.use_llm else ExtractiveAnswerGenerator()
    answer = generator.generate(req.question, results)

    return {"answer": answer, "results": results}





@app.post("/ask_ui", response_class=HTMLResponse)
def ask_ui(request: Request, question: str = Form(...), use_llm: str = Form(None)):
    """
    Handle UI form submission for questions.
    """
    global retriever
    if retriever is None:
        raise HTTPException(status_code=400, detail="Index not found. Upload a PDF first.")

    q_ctx = memory.contextualize("ui", question)
    results = retriever.search(q_ctx, settings.TOP_K)
    topic = memory.infer_topic(question)
    memory.update("ui", question, topic)

    generator: AnswerGenerator = LLMAnswerGenerator() if use_llm else ExtractiveAnswerGenerator()
    answer = generator.generate(question, results)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": answer,
        "results": results
    })
