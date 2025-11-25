# file: src/assistant/service/api.py

from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import warnings
import json
import base64
import zlib
from datetime import datetime, timedelta

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
    MedicationItem,
    AppointmentItem,
    DoctorNote,
)

# --- Phase 6 security + QR imports ---
from assistant.security.crypto import encrypt_bytes, decrypt_bytes
from assistant.share.qr_utils import make_qr_png_base64

# --- Phase 7 ICS helper ---
from assistant.share.ics_utils import build_ics_event

# ---- FastAPI app ----
app = FastAPI(title="Private Health LLM", version="0.7.0 - Phase6+7 Integrations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared model client
llm = OllamaClient(model="mistral:latest", timeout_s=600)

SYSTEM_PROMPT = (
    "You are a careful health assistant. Be concise, factual, and supportive. "
    "You are not a doctor and do not provide diagnosis; you provide general information "
    "and encourage users to consult their clinician for personal medical advice."
)

# -----------------------------------------
# System Health
# -----------------------------------------
@app.get("/health", response_model=Health, tags=["System"])
def health() -> Health:
    return Health(status="ok")


# -----------------------------------------
# Ask Endpoint
# -----------------------------------------
@app.post("/ask", response_model=AskRes, tags=["Ask"])
def ask(req: AskReq) -> AskRes:
    user_id = req.user_id or "demo"
    recent = mem.format_for_context(user_id, limit=8)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent + [
        {"role": "user", "content": req.question}
    ]

    try:
        answer = llm.chat(messages, temperature=0.1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e!r}")

    mem.add(user_id=user_id, role="user", content=req.question)
    mem.add(user_id=user_id, role="assistant", content=answer)

    DigitalTwinManager(user_id).log_event("ask_answer", content=answer, source=f"ask:{req.question}")
    return AskRes(answer=answer)


# -----------------------------------------
# Memory Endpoints
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
# Ingestion & Summarization
# -------------------------------------------------------------
@app.post("/ingest", tags=["Ingest"])
async def ingest_file_api(file: UploadFile = File(...), user_id: str = Query(default="demo")):
    file_bytes = await file.read()
    filename = file.filename or "upload"
    file_type = "pdf" if filename.lower().endswith(".pdf") else "image"

    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!r}")

    append_ingested_text(user_id, extracted_text, source=filename)

    try:
        summary = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this medical report:\n\n{extracted_text}"},
            ]
        )
        DigitalTwinManager(user_id).log_event("summary", summary, f"ingest:{filename}")
    except Exception as e:
        summary = "Summary generation failed."
        print(f"⚠️ LLM summary failed: {e}")

    alert_keywords = ["abnormal", "high", "low", "elevated", "critical"]
    alert_triggered = any(word in extracted_text.lower() for word in alert_keywords)

    return {
        "status": "success",
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "summary": summary,
        "alert": alert_triggered,
    }


# -------------------------------------------------------------
# Master Upload Endpoint
# -------------------------------------------------------------
@app.post("/upload-document", tags=["Master Ingestion"])
async def upload_document_master(
    file: UploadFile = File(...), user_id: str = Query(default="demo")
) -> Dict[str, Any]:
    file_bytes = await file.read()
    filename = file.filename or "upload"
    file_type = "pdf" if filename.lower().endswith(".pdf") else "image"

    try:
        extracted_text = ingest_file(file_bytes, file_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial ingestion failed: {e!r}")

    append_ingested_text(user_id, extracted_text, source=filename)
    doc_type = classify_document(extracted_text)
    structured_data: Dict[str, Any] = {}

    if doc_type != "general_text" and file_type == "image":
        questions = get_question_set(doc_type)
        try:
            structured_data = extract_answers_from_image_bytes(file_bytes, questions)
        except Exception as e:
            warnings.warn(f"Donut extraction failed for {doc_type}. Error: {e}")

    try:
        summary = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Summarize this document (classified as {doc_type}):\n\n{extracted_text}",
                },
            ]
        )
        mem.add(user_id=user_id, role="assistant", content=summary)
        DigitalTwinManager(user_id).log_event(
            "summary", summary, f"upload:{filename}|type:{doc_type}"
        )
    except Exception:
        summary = "Summary generation failed."

    return {"status": "success", "doc_type": doc_type, "summary": summary, "structured_data": structured_data}


# -------------------------------------------------------------
# Donut Parser Endpoint
# -------------------------------------------------------------
@app.post("/parse-lab-report", tags=["CV"])
async def parse_lab_report(file: UploadFile = File(...), user_id: str = Query(default="demo")):
    file_bytes = await file.read()
    filename = file.filename or "upload"

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only images are supported.")

    extracted_text = ingest_file(file_bytes, file_type="image")
    doc_type = classify_document(extracted_text)
    questions = get_question_set(doc_type)

    parsed_answers: Dict[str, Any] = {}
    if doc_type != "general_text":
        try:
            parsed_answers = extract_answers_from_image_bytes(file_bytes, questions)
        except Exception as e:
            warnings.warn(f"Donut failed. Error: {e}")
            summary = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Summarize this document ({doc_type}):\n\n{extracted_text}",
                    },
                ]
            )
            return {"status": "fallback_summary", "summary": summary}

    return {"status": "success", "doc_type": doc_type, "parsed_answers": parsed_answers}


# -------------------------------------------------------------
# Digital Twin Expansion
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
    twin = DigitalTwinManager(req.user_id).add_user_note(req.text)
    return {"status": "success", "twin": twin.model_dump()}


@app.post("/add-health-history", tags=["Digital Twin"])
def add_health_history(req: HealthHistoryRequest):
    twin = DigitalTwinManager(req.user_id).add_health_history(req.items)
    return {"status": "success", "twin": twin.model_dump()}


@app.post("/add-surgeries", tags=["Digital Twin"])
def add_surgeries(req: SurgeryRequest):
    twin = DigitalTwinManager(req.user_id).add_surgeries(req.items)
    return {"status": "success", "twin": twin.model_dump()}


@app.post("/add-family-history", tags=["Digital Twin"])
def add_family_history(req: FamilyHistoryRequest):
    twin = DigitalTwinManager(req.user_id).add_family_history(req.items)
    return {"status": "success", "twin": twin.model_dump()}


@app.post("/suggest-followup-questions", tags=["Digital Twin"])
def suggest_followup(req: FollowupRequest):
    dt = DigitalTwinManager(req.user_id)
    context = dt.build_context()
    questions = llm.generate_followup_questions(req.summary_or_text, context)
    dt.log_event("followup_questions", content=questions, source="suggest-followup")
    return {"status": "success", "questions": questions}


# -------------------------------------------------------------
# Predictive Modeling
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
    return {"status": "success", "twin": twin.model_dump()}


@app.post("/predict-risk", tags=["Predictive Modeling"])
def predict_risk(req: PredictionRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.twin
    preds: List[RiskPrediction] = []

    for cond in req.conditions:
        score = 0.3
        if any(f.condition.lower() == cond.lower() for f in twin.family_history):
            score += 0.2
        if any(h.condition.lower() == cond.lower() for h in twin.health_history):
            score += 0.3
        if twin.wearables:
            avg_hr = sum(w.heart_rate or 0 for w in twin.wearables) / len(twin.wearables)
            if cond.lower() == "hypertension" and avg_hr > 85:
                score += 0.2
        score = min(1.0, score)

        pred = RiskPrediction(condition=cond, risk_score=score, explanation=f"Risk for {cond} computed.")
        twin.predictions.append(pred)
        preds.append(pred)

    save_twin(twin)
    DigitalTwinManager(req.user_id).log_event(
        "prediction", content=str([p.model_dump() for p in preds]), source="predict-risk"
    )
    return {"status": "success", "predictions": [p.model_dump() for p in preds]}


@app.get("/forecast-wearables", tags=["Predictive Modeling"])
def forecast_wearables(user_id: str):
    dt = DigitalTwinManager(user_id)
    twin = dt.twin
    if not twin.wearables:
        raise HTTPException(status_code=400, detail="No wearable data available.")
    avg_steps = sum(w.steps or 0 for w in twin.wearables) / len(twin.wearables)
    avg_hr = sum(w.heart_rate or 0 for w in twin.wearables) / len(twin.wearables)
    avg_sleep = sum(w.sleep_hours or 0 for w in twin.wearables) / len(twin.wearables)
    forecast = {
        "steps": round(avg_steps),
        "heart_rate": round(avg_hr, 1),
        "sleep_hours": round(avg_sleep, 1),
    }
    DigitalTwinManager(user_id).log_event("prediction", content=str(forecast), source="forecast-wearables")
    return {"status": "success", "forecast": forecast}


# -------------------------------------------------------------
# PHASE 6: Digital Twin + QR (Encrypted + Compressed)
# -------------------------------------------------------------
class GenerateQRRequest(BaseModel):
    user_id: str
    include_notes: bool = False
    include_history: bool = True
    include_family: bool = True
    include_surgeries: bool = False
    include_ingested_items: bool = False
    include_wearables: bool = True
    include_predictions: bool = True
    expires_in_minutes: int = 10
    scope_label: str = "clinician_view"


class GenerateQRResponse(BaseModel):
    status: str
    user_id: str
    token_b64: str
    qr_png_base64: str
    expires_at: str
    scope_label: str


def _scoped_twin_view(twin, cfg: GenerateQRRequest) -> dict:
    out = {"user_id": twin.user_id}
    if cfg.include_history:
        out["health_history"] = [h.model_dump() for h in twin.health_history[-12:]]
    if cfg.include_family:
        out["family_history"] = [f.model_dump() for f in twin.family_history[-12:]]
    if cfg.include_surgeries:
        out["surgeries"] = [s.model_dump() for s in twin.surgeries[-8:]]
    if cfg.include_notes:
        out["notes"] = [n.model_dump() for n in twin.notes[-4:]]
    if cfg.include_ingested_items:
        out["ingested_items"] = [
            {**i.model_dump(), "text": (i.text[:500] + "...") if len(i.text) > 500 else i.text}
            for i in twin.ingested_items[-3:]
        ]
    if cfg.include_wearables:
        out["wearables"] = [w.model_dump() for w in twin.wearables[-7:]]
    if cfg.include_predictions:
        out["predictions"] = [p.model_dump() for p in twin.predictions[-7:]]
    return out


@app.post("/generate-dht-qr", tags=["Digital Twin", "QR"], response_model=GenerateQRResponse)
def generate_dht_qr(req: GenerateQRRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.snapshot()
    scoped = _scoped_twin_view(twin, req)
    issued_at = datetime.utcnow()
    expires_at = issued_at + timedelta(minutes=max(1, req.expires_in_minutes))

    envelope = {
        "user_id": req.user_id,
        "scope": req.scope_label,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "data": scoped,
    }

    try:
        plaintext = json.dumps(envelope, default=str).encode("utf-8")
        compressed = zlib.compress(plaintext)
        ciphertext = encrypt_bytes(compressed)
        token_b64 = base64.urlsafe_b64encode(ciphertext).decode("utf-8")
        qr_png_base64 = make_qr_png_base64(token_b64)
        dt.log_event("qr", f"QR generated scope={req.scope_label}", "generate-dht-qr")
        return GenerateQRResponse(
            status="success",
            user_id=req.user_id,
            token_b64=token_b64,
            qr_png_base64=qr_png_base64,
            expires_at=expires_at.isoformat(),
            scope_label=req.scope_label,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QR generation failed: {e}")


class DecodeQRRequest(BaseModel):
    token_b64: str


@app.post("/decode-dht-token", tags=["Digital Twin", "QR"])
def decode_dht_token(req: DecodeQRRequest):
    try:
        ciphertext = base64.urlsafe_b64decode(req.token_b64.encode("utf-8"))
        decrypted = decrypt_bytes(ciphertext)
        decompressed = zlib.decompress(decrypted)
        envelope = json.loads(decompressed.decode("utf-8"))
        exp = datetime.fromisoformat(envelope.get("expires_at", ""))
        if datetime.utcnow() > exp:
            raise HTTPException(status_code=410, detail="Token expired.")
        user_id = envelope.get("user_id", "unknown")
        DigitalTwinManager(user_id).log_event("qr", f"QR decoded scope={envelope.get('scope')}", "decode-dht-token")
        return {"status": "success", "envelope": envelope}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token decoding failed: {e}")


# -------------------------------------------------------------
# PHASE 7: Medications, Interactions, Appointments, Notes
# -------------------------------------------------------------
# ---- Medications ----
class MedicationCreateRequest(BaseModel):
    user_id: str
    medications: List[MedicationItem]


class MedicationInteractionCheckRequest(BaseModel):
    user_id: str
    medications: List[str]


class MedicationInteractionCheckResponse(BaseModel):
    status: str
    user_id: str
    interactions: List[Dict[str, str]]


# Very simple static interaction examples (for demo only, NOT medical advice)
KNOWN_INTERACTIONS = [
    ({"ibuprofen", "aspirin"}, "Ibuprofen and aspirin together may increase bleeding risk."),
    ({"metformin", "contrast dye"}, "Metformin with contrast dye may increase risk of lactic acidosis."),
    ({"warfarin", "aspirin"}, "Warfarin and aspirin together may significantly increase bleeding risk."),
]


@app.post("/add-medications", tags=["Medications"])
def add_medications(req: MedicationCreateRequest):
    dt = DigitalTwinManager(req.user_id)
    twin = dt.twin
    twin.medications.extend(req.medications)
    save_twin(twin)
    dt.log_event("medication", content=f"Added {len(req.medications)} meds", source="add-medications")
    return {"status": "success", "user_id": req.user_id, "twin": twin.model_dump()}


@app.post("/check-medication-interactions", tags=["Medications"], response_model=MedicationInteractionCheckResponse)
def check_medication_interactions(req: MedicationInteractionCheckRequest):
    meds_lower = {m.lower().strip() for m in req.medications}
    found: List[Dict[str, str]] = []

    for combo, msg in KNOWN_INTERACTIONS:
        if combo.issubset(meds_lower):
            found.append(
                {
                    "pair": ", ".join(sorted(combo)),
                    "message": msg,
                }
            )

    DigitalTwinManager(req.user_id).log_event(
        "medication",
        content=f"Interaction check for meds={req.medications} -> {len(found)} hits",
        source="check-medication-interactions",
    )
    return MedicationInteractionCheckResponse(status="success", user_id=req.user_id, interactions=found)


# ---- Appointment booking + ICS ----
class AppointmentRequest(BaseModel):
    user_id: str
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    notes: Optional[str] = None


class AppointmentICSResponse(BaseModel):
    status: str
    user_id: str
    appointment_id: str
    ics_base64: str


@app.post("/book-appointment", tags=["Ecosystem", "Appointments"], response_model=AppointmentICSResponse)
def book_appointment(req: AppointmentRequest):
    if req.end_time <= req.start_time:
        raise HTTPException(status_code=400, detail="end_time must be after start_time.")

    dt = DigitalTwinManager(req.user_id)
    twin = dt.twin

    appt_id = f"appt-{len(twin.appointments) + 1}-{int(datetime.utcnow().timestamp())}"

    appt = AppointmentItem(
        id=appt_id,
        title=req.title,
        start_time=req.start_time,
        end_time=req.end_time,
        location=req.location,
        notes=req.notes,
    )
    twin.appointments.append(appt)
    save_twin(twin)

    ics_bytes = build_ics_event(
        uid=appt_id,
        title=req.title,
        start=req.start_time,
        end=req.end_time,
        location=req.location or "",
        description=req.notes or "",
    )
    ics_b64 = base64.b64encode(ics_bytes).decode("utf-8")

    dt.log_event(
        "appointment",
        content=f"Booked {req.title} from {req.start_time.isoformat()} to {req.end_time.isoformat()}",
        source="book-appointment",
    )

    return AppointmentICSResponse(
        status="success",
        user_id=req.user_id,
        appointment_id=appt_id,
        ics_base64=ics_b64,
    )


# ---- Doctor / Therapist note summarization ----
class DoctorNoteUploadRequest(BaseModel):
    user_id: str
    raw_text: str
    visit_type: Optional[str] = "doctor"


class DoctorNoteResponse(BaseModel):
    status: str
    user_id: str
    summary: str
    bullets: List[str]


@app.post("/summarize-doctor-note", tags=["Ecosystem", "Notes"], response_model=DoctorNoteResponse)
def summarize_doctor_note(req: DoctorNoteUploadRequest):
    dt = DigitalTwinManager(req.user_id)
    context = dt.build_context()

    prompt = (
        "You are a health assistant helping a patient understand their visit notes.\n\n"
        "1) Read the following raw note.\n"
        "2) Produce a short plain-language summary.\n"
        "3) Extract 3–7 key action points as bullets (medications, follow-ups, lifestyle advice).\n\n"
        f"Patient context (from digital twin):\n{context}\n\n"
        f"Raw note:\n{req.raw_text}\n\n"
        "Return output in this format:\n"
        "SUMMARY:\n...\n\nBULLETS:\n- ...\n- ...\n"
    )

    try:
        llm_output = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM summarization failed: {e!r}")

    # Very simple parsing: split into summary + bullets
    summary_text = llm_output
    bullets: List[str] = []
    if "BULLETS:" in llm_output:
        parts = llm_output.split("BULLETS:", 1)
        summary_text = parts[0].replace("SUMMARY:", "").strip()
        bullets_raw = parts[1]
        bullets = [line.strip(" -•") for line in bullets_raw.splitlines() if line.strip().startswith(("-", "•"))]

    twin = dt.twin
    note = DoctorNote(raw_text=req.raw_text, summary=summary_text, visit_type=req.visit_type)
    twin.doctor_notes.append(note)
    save_twin(twin)

    dt.log_event("note_summary", content=summary_text, source="summarize-doctor-note")

    return DoctorNoteResponse(
        status="success",
        user_id=req.user_id,
        summary=summary_text,
        bullets=bullets,
    )
