# file: src/assistant/service/api.py

from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import warnings

from assistant.llm.ollama_client import OllamaClient
from assistant.memory.store import mem
from assistant.schema import AskReq, AskRes, MemoryList, MemoryItem, Health
from assistant.ingest.file_ingestor import ingest_file
# --- NEW/UPDATED IMPORTS ---
from assistant.ocr.donut_parser import extract_answers_from_image_bytes
from assistant.ingest.document_router import classify_document, get_question_set

# --- Digital Twin imports ---
from pydantic import BaseModel
from typing import List
from assistant.memory.digital_twin import DigitalTwinManager
from assistant.schema.dht_models import HealthHistoryItem, FamilyHistoryItem
from assistant.llm.ollama_client import OllamaClient

# ---- FastAPI app ----
app = FastAPI(title="Private Health LLM", version="0.4.1 - Robust Routing")

# Basic CORS so you can call this from a local web app later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared client to the local model
llm = OllamaClient(model="mistral:latest", timeout_s=600)

SYSTEM_PROMPT = (
    "You are a careful health assistant. Be concise, factual, and supportive. "
    "You are not a doctor and do not provide diagnosis; you provide general information "
    "and encourage users to consult their clinician for personal medical advice."
)

# -----------------------------------------
# System health endpoint
# -----------------------------------------
@app.get("/health", response_model=Health, tags=["System"])
def health() -> Health:
    return Health(status="ok")

# -----------------------------------------
# Ask LLM endpoint
# -----------------------------------------
@app.post("/ask", response_model=AskRes, tags=["Ask"])
def ask(req: AskReq) -> AskRes:
    user_id = req.user_id or "demo"

    # 1) Build messages
    recent = mem.format_for_context(user_id, limit=8)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent)
    messages.append({"role": "user", "content": req.question})

    # 2) Call LLM
    try:
        answer = llm.chat(messages, temperature=0.1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e!r}")

    # 3) Store in memory
    mem.add(user_id=user_id, role="user", content=req.question)
    mem.add(user_id=user_id, role="assistant", content=answer)

    return AskRes(answer=answer)

# -----------------------------------------
# Memory endpoints
# -----------------------------------------
@app.get("/memory/{user_id}", response_model=MemoryList, tags=["Memory"])
def get_memory(user_id: str, limit: int = 20) -> MemoryList:
    items = [MemoryItem(**m) for m in mem.get(user_id, limit=limit)]
    return MemoryList(user_id=user_id, items=items)

@app.delete("/memory/{user_id}", tags=["Memory"])
def clear_memory(user_id: str):
    mem._store[user_id].clear()  # type: ignore[attr-defined]
    return {"status": "cleared", "user_id": user_id}

# -------------------------------------------------------------
# RESTORED ENDPOINT: General OCR & LLM Summarization (The Fallback)
# -------------------------------------------------------------
@app.post("/ingest", tags=["Ingest"])
async def ingest_file_api(
    file: UploadFile = File(...),
    user_id: str = Query(default="demo")
):
    file_bytes = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        file_type = "pdf"
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        file_type = "image"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!r}")

    # 🧠 Store raw extracted text in memory
    mem.add(user_id=user_id, role="user", content=f"[Uploaded {file_type}] {extracted_text}")

    # 🤖 Generate auto-summary via LLM
    try:
        summary = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this medical report:\n\n{extracted_text}"}
        ])
    except Exception as e:
        summary = "Summary generation failed."
        print(f"⚠️ LLM summary failed: {e}")

    # 🧠 Store summary in memory
    mem.add(user_id=user_id, role="assistant", content=summary)

    # 🚨 Basic alert logic: flag if abnormalities are mentioned
    alert_keywords = ["abnormal", "high", "low", "elevated", "critical"]
    alert_triggered = any(word in extracted_text.lower() for word in alert_keywords)

    return {
        "status": "success",
        "user_id": user_id,
        "filename": file.filename,
        "file_type": file_type,
        "extracted_text": extracted_text,
        "summary": summary,
        "alert": alert_triggered
    }

# -------------------------------------------------------------
# MASTER ENDPOINT: Dynamic Routing & Extraction
# -------------------------------------------------------------
@app.post("/upload-document", tags=["Master Ingestion"])
async def upload_document_master(
    file: UploadFile = File(...),
    user_id: str = Query(default="demo")
) -> Dict[str, Any]:
    file_bytes = await file.read()
    filename = file.filename.lower()
    file_type = "pdf" if filename.endswith(".pdf") else "image"

    # 1. Initial OCR (Raw Text Extraction) for Classification Input
    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial ingestion failed: {e!r}")

    # 2. Dynamic Routing (Classification)
    doc_type = classify_document(extracted_text)
    structured_data = {}

    # 3. Attempt Structured Extraction (only for images of specific types)
    if doc_type != "general_text" and file_type == "image":
        questions = get_question_set(doc_type)
        try:
            structured_data = extract_answers_from_image_bytes(file_bytes, questions)
        except Exception as e:
            warnings.warn(f"Donut extraction failed for {doc_type}. Error: {e}")

    # 4. LLM Summarization and Memory Storage
    try:
        summary = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this document (classified as {doc_type}):\n\n{extracted_text}"}
        ])
        mem.add(user_id=user_id, role="user", content=f"[Uploaded {doc_type}] {extracted_text}")
        mem.add(user_id=user_id, role="assistant", content=summary)
    except Exception:
        summary = "Summary generation failed."

    return {
        "status": "success",
        "doc_type": doc_type,
        "llm_summary": summary,
        "structured_data": structured_data,
    }

# -------------------------------------------------------------
# RESTORED ENDPOINT: Specialized Donut Endpoint
# -------------------------------------------------------------
@app.post("/parse-lab-report", tags=["CV"])
async def parse_lab_report(
    file: UploadFile = File(...),
    user_id: str = Query(default="demo")
):
    file_bytes = await file.read()
    filename = file.filename.lower()

    if not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only images are supported.")

    # 1. Classify the document type
    try:
        extracted_text = ingest_file(file_bytes, file_type="image")
        doc_type = classify_document(extracted_text)
        questions = get_question_set(doc_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial classification failed: {e!r}")

    # 2. Attempt Structured Extraction
    parsed_answers = {}
    if doc_type != "general_text":
        try:
            parsed_answers = extract_answers_from_image_bytes(file_bytes, questions)
        except Exception as e:
            warnings.warn(f"Donut failed. Falling back. Error: {e}")
            try:
                summary = llm.chat([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Summarize this document (classified as {doc_type}):\n\n{extracted_text}"}
                ])
                return {
                    "status": "fallback_summary",
                    "doc_type": doc_type,
                    "summary": summary,
                }
            except Exception:
                raise HTTPException(status_code=500, detail="Both structured and summary failed.")

    return {
        "status": "success",
        "user_id": user_id,
        "doc_type": doc_type,
        "filename": file.filename,
        "parsed_answers": parsed_answers,
    }

# -------------------------------------------------------------
# NEW ENDPOINTS: Digital Twin Expansion
# -------------------------------------------------------------
class UserNoteRequest(BaseModel):
    user_id: str
    text: str

class HealthHistoryRequest(BaseModel):
    user_id: str
    items: List[HealthHistoryItem]

class FamilyHistoryRequest(BaseModel):
    user_id: str
    items: List[FamilyHistoryItem]

class FollowupRequest(BaseModel):
    user_id: str
    summary_or_text: str

@app.post("/add-user-note", tags=["Digital Twin"])
def add_user_note(req: UserNoteRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.add_user_note(req.text)
    return {"status": "success", "user_id": req.user_id, "twin": twin.model_dump()}

@app.post("/add-health-history", tags=["Digital Twin"])
def add_health_history(req: HealthHistoryRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.add_health_history(req.items)
    return {"status": "success", "user_id": req.user_id, "twin": twin.model_dump()}

@app.post("/add-family-history", tags=["Digital Twin"])
def add_family_history(req: FamilyHistoryRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.add_family_history(req.items)
    return {"status": "success", "user_id": req.user_id, "twin": twin.model_dump()}

@app.post("/suggest-followup-questions", tags=["Digital Twin"])
def suggest_followup(req: FollowupRequest):
    dt = DigitalTwinManager(req.user_id)
    context = dt.build_context()
    try:
        questions = llm.generate_followup_questions(req.summary_or_text, context)
        return {
            "status": "success",
            "user_id": req.user_id,
            "context_used": context,
            "questions": questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
