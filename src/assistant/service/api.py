# file: src/assistant/service/api.py

from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import warnings

from pydantic import BaseModel

# --- Core imports ---
from assistant.llm.ollama_client import OllamaClient
from assistant.memory.store import mem, append_ingested_text, save_twin
from assistant.schema import AskReq, AskRes, MemoryList, MemoryItem, Health
from assistant.ingest.file_ingestor import ingest_file

# --- Phase 3 OCR/CV imports ---
from assistant.ocr.donut_parser import extract_answers_from_image_bytes
from assistant.ingest.document_router import classify_document, get_question_set

# --- Digital Twin imports ---
from assistant.memory.digital_twin import DigitalTwinManager
from assistant.schema.dht_models import (
    HealthHistoryItem,
    FamilyHistoryItem,
    SurgeryItem,
    WearableRecord,
    RiskPrediction,
)

# ---- FastAPI app ----
app = FastAPI(title="Private Health LLM", version="0.6.0 - Phase5 Predictive Modeling")

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
# Ask LLM endpoint (logs to DHT)
# -----------------------------------------
@app.post("/ask", response_model=AskRes, tags=["Ask"])
def ask(req: AskReq) -> AskRes:
    user_id = req.user_id or "demo"

    recent = mem.format_for_context(user_id, limit=8)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent)
    messages.append({"role": "user", "content": req.question})

    try:
        answer = llm.chat(messages, temperature=0.1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e!r}")

    mem.add(user_id=user_id, role="user", content=req.question)
    mem.add(user_id=user_id, role="assistant", content=answer)

    dt = DigitalTwinManager(user_id)
    dt.log_event(response_type="ask_answer", content=answer, source=f"ask:{req.question}")

    return AskRes(answer=answer)

# -----------------------------------------
# Memory endpoints (legacy)
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
# General OCR & LLM Summarization + DHT logging
# -------------------------------------------------------------
@app.post("/ingest", tags=["Ingest"])
async def ingest_file_api(file: UploadFile = File(...), user_id: str = Query(default="demo")):
    file_bytes = await file.read()
    filename = file.filename or "upload"
    filename_l = filename.lower()

    if filename_l.endswith(".pdf"):
        file_type = "pdf"
    elif filename_l.endswith((".png", ".jpg", ".jpeg")):
        file_type = "image"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!r}")

    append_ingested_text(user_id, extracted_text, source=filename)

    try:
        summary = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this medical report:\n\n{extracted_text}"}
        ])
        DigitalTwinManager(user_id).log_event(
            response_type="summary",
            content=summary,
            source=f"ingest:{filename}"
        )
    except Exception as e:
        summary = "Summary generation failed."
        print(f"⚠️ LLM summary failed: {e}")

    mem.add(user_id=user_id, role="user", content=f"[Uploaded {file_type}] {extracted_text}")
    mem.add(user_id=user_id, role="assistant", content=summary)

    alert_keywords = ["abnormal", "high", "low", "elevated", "critical"]
    alert_triggered = any(word in extracted_text.lower() for word in alert_keywords)

    return {
        "status": "success",
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "extracted_text": extracted_text,
        "summary": summary,
        "alert": alert_triggered
    }

# -------------------------------------------------------------
# MASTER ENDPOINT: Dynamic Routing & Extraction + DHT logging
# -------------------------------------------------------------
@app.post("/upload-document", tags=["Master Ingestion"])
async def upload_document_master(file: UploadFile = File(...), user_id: str = Query(default="demo")) -> Dict[str, Any]:
    file_bytes = await file.read()
    filename = file.filename or "upload"
    filename_l = filename.lower()
    file_type = "pdf" if filename_l.endswith(".pdf") else "image"

    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial ingestion failed: {e!r}")

    append_ingested_text(user_id, extracted_text, source=filename)

    doc_type = classify_document(extracted_text)
    structured_data = {}

    if doc_type != "general_text" and file_type == "image":
        questions = get_question_set(doc_type)
        try:
            structured_data = extract_answers_from_image_bytes(file_bytes, questions)
        except Exception as e:
            warnings.warn(f"Donut extraction failed for {doc_type}. Error: {e}")

    try:
        summary = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this document (classified as {doc_type}):\n\n{extracted_text}"}
        ])
        mem.add(user_id=user_id, role="user", content=f"[Uploaded {doc_type}] {extracted_text}")
        mem.add(user_id=user_id, role="assistant", content=summary)

        DigitalTwinManager(user_id).log_event(
            response_type="summary",
            content=summary,
            source=f"upload:{filename}|type:{doc_type}"
        )
    except Exception:
        summary = "Summary generation failed."

    return {
        "status": "success",
        "doc_type": doc_type,
        "llm_summary": summary,
        "structured_data": structured_data,
    }

# -------------------------------------------------------------
# Specialized Donut Endpoint + DHT logging
# -------------------------------------------------------------
@app.post("/parse-lab-report", tags=["CV"])
async def parse_lab_report(file: UploadFile = File(...), user_id: str = Query(default="demo")):
    file_bytes = await file.read()
    filename = file.filename or "upload"
    filename_l = filename.lower()

    if not filename_l.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only images are supported.")

    try:
        extracted_text = ingest_file(file_bytes, file_type="image")
        doc_type = classify_document(extracted_text)
        questions = get_question_set(doc_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial classification failed: {e!r}")

    append_ingested_text(user_id, extracted_text, source=filename)

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
                DigitalTwinManager(user_id).log_event(
                    response_type="summary",
                    content=summary,
                    source=f"parse-lab-report:{filename}|type:{doc_type}"
                )
                return {"status": "fallback_summary", "doc_type": doc_type, "summary": summary}
            except Exception:
                raise HTTPException(status_code=500, detail="Both structured and summary failed.")

    return {
        "status": "success",
        "user_id": user_id,
        "doc_type": doc_type,
        "filename": filename,
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

class SurgeryRequest(BaseModel):
    user_id: str
    items: List[SurgeryItem]

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

@app.post("/add-surgeries", tags=["Digital Twin"])
def add_surgeries(req: SurgeryRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.add_surgeries(req.items)
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
        dt.log_event("followup_questions", content=questions, source="suggest-followup-questions")
        return {"status": "success", "user_id": req.user_id, "context_used": context, "questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------
# PHASE 5: Predictive Modeling Endpoints
# -------------------------------------------------------------
class WearableRequest(BaseModel):
    user_id: str
    records: List[WearableRecord]

class PredictionRequest(BaseModel):
    user_id: str
    conditions: List[str]

@app.post("/add-wearable-data", tags=["Predictive Modeling"])
def add_wearable_data(req: WearableRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.twin
    twin.wearables.extend(req.records)
    save_twin(twin)
    return {"status": "success", "user_id": req.user_id, "twin": twin.model_dump()}

@app.post("/predict-risk", tags=["Predictive Modeling"])
def predict_risk(req: PredictionRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.twin

    predictions: List[RiskPrediction] = []
    for cond in req.conditions:
        base_score = 0.3
        if any(f.condition.lower() == cond.lower() for f in twin.family_history):
            base_score += 0.2
        if any(h.condition.lower() == cond.lower() for h in twin.health_history):
            base_score += 0.3
        if twin.wearables:
            avg_hr = sum(w.heart_rate or 0 for w in twin.wearables) / len(twin.wearables)
            if cond.lower() == "hypertension" and avg_hr > 85:
                base_score += 0.2
        score = min(1.0, base_score)

        pred = RiskPrediction(condition=cond, risk_score=score,
                              explanation=f"Computed risk for {cond} based on history/family/wearables.")
        twin.predictions.append(pred)
        predictions.append(pred)

    save_twin(twin)
    dt.log_event("prediction", content=str([p.model_dump() for p in predictions]), source="predict-risk")

    return {"status": "success", "user_id": req.user_id, "predictions": [p.model_dump() for p in predictions]}

@app.get("/forecast-wearables", tags=["Predictive Modeling"])
def forecast_wearables(user_id: str):
    dt = DigitalTwinManager(user_id)
    twin = dt.twin
    if not twin.wearables:
        raise HTTPException(status_code=400, detail="No wearable data available.")

    avg_steps = sum(w.steps or 0 for w in twin.wearables) / len(twin.wearables)
    avg_hr = sum(w.heart_rate or 0 for w in twin.wearables) / len(twin.wearables)
    avg_sleep = sum(w.sleep_hours or 0 for w in twin.wearables) / len(twin.wearables)

    forecast = {"steps": round(avg_steps), "heart_rate": round(avg_hr, 1), "sleep_hours": round(avg_sleep, 1)}

    dt.log_event("prediction", content=str(forecast), source="forecast-wearables")

    return {"status": "success", "user_id": user_id, "forecast": forecast}
