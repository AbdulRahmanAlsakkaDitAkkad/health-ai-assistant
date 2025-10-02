from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from assistant.llm.ollama_client import OllamaClient
from assistant.memory.store import mem
from assistant.schema import AskReq, AskRes, MemoryList, MemoryItem, Health
from assistant.ingest.file_ingestor import ingest_file

# ---- FastAPI app ----
app = FastAPI(title="Private Health LLM", version="0.2.0")

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

@app.get("/health", response_model=Health, tags=["System"])
def health() -> Health:
    return Health(status="ok")

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

@app.get("/memory/{user_id}", response_model=MemoryList, tags=["Memory"])
def get_memory(user_id: str, limit: int = 20) -> MemoryList:
    items = [MemoryItem(**m) for m in mem.get(user_id, limit=limit)]
    return MemoryList(user_id=user_id, items=items)

@app.delete("/memory/{user_id}", tags=["Memory"])
def clear_memory(user_id: str):
    mem._store[user_id].clear()  # type: ignore[attr-defined]
    return {"status": "cleared", "user_id": user_id}

@app.post("/ingest", tags=["Ingest"])
async def ingest_file_api(
    file: UploadFile = File(...),
    user_id: str = Query(default="demo")
):
    """
    Upload a PDF or image, extract its text via OCR, summarize it, store in memory, and return metadata.
    """
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
