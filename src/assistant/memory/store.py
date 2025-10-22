# src/assistant/memory/store.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable
from pydantic import ValidationError

from assistant.schema.dht_models import (
    DigitalHealthTwin,
    IngestedItem,
    LogEntry,
    UserNote,
    HealthHistoryItem,
    FamilyHistoryItem,
    SurgeryItem,
)

# -------------------------------------------------------------------
# Digital Twin persistence (JSON per user in ./data/twins/{user_id}.json)
# -------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "twins"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _user_path(user_id: str) -> Path:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return DATA_DIR / f"{safe}.json"

def load_twin(user_id: str) -> DigitalHealthTwin:
    p = _user_path(user_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return DigitalHealthTwin.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            # file corrupted -> start fresh
            return DigitalHealthTwin(user_id=user_id)
    return DigitalHealthTwin(user_id=user_id)

def save_twin(twin: DigitalHealthTwin) -> None:
    p = _user_path(twin.user_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(twin.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)

# ----------------------- Append helpers -----------------------------

def append_ingested_text(user_id: str, text: str, source: str) -> None:
    twin = load_twin(user_id)
    twin.ingested_items.append(IngestedItem(source=source, text=text))
    save_twin(twin)

def append_log(
    user_id: str,
    response_type: LogEntry.__fields__['response_type'].annotation,  # Literal types
    content: str,
    source: str | None = None,
) -> None:
    twin = load_twin(user_id)
    twin.logs.append(LogEntry(response_type=response_type, content=content, source=source))
    save_twin(twin)

# ----------------------- Validation / de-dup ------------------------

def _lower(s: str | None) -> str:
    return (s or "").strip().lower()

def dedupe_health_history(existing: list[HealthHistoryItem], new_items: Iterable[HealthHistoryItem]) -> list[HealthHistoryItem]:
    seen = {( _lower(e.condition), _lower(e.diagnosis_date) ) for e in existing}
    result = list(existing)
    for it in new_items:
        key = (_lower(it.condition), _lower(it.diagnosis_date))
        if key not in seen:
            result.append(it)
            seen.add(key)
    return result

def dedupe_family_history(existing: list[FamilyHistoryItem], new_items: Iterable[FamilyHistoryItem]) -> list[FamilyHistoryItem]:
    seen = {( _lower(e.relation), _lower(e.condition) ) for e in existing}
    result = list(existing)
    for it in new_items:
        key = (_lower(it.relation), _lower(it.condition))
        if key not in seen:
            result.append(it)
            seen.add(key)
    return result

def dedupe_surgeries(existing: list[SurgeryItem], new_items: Iterable[SurgeryItem]) -> list[SurgeryItem]:
    seen = {( _lower(e.name), _lower(e.date) ) for e in existing}
    result = list(existing)
    for it in new_items:
        key = (_lower(it.name), _lower(it.date))
        if key not in seen:
            result.append(it)
            seen.add(key)
    return result

def dedupe_notes(existing: list[UserNote], new_items: Iterable[UserNote]) -> list[UserNote]:
    # match on text only (keep latest timestamp for identical text)
    seen_texts = { _lower(e.text) for e in existing }
    result = list(existing)
    for it in new_items:
        if _lower(it.text) not in seen_texts:
            result.append(it)
            seen_texts.add(_lower(it.text))
    return result

# -------------------------------------------------------------------
# Legacy in-memory store (for backward compatibility with /ask memory)
# -------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        self._store: dict[str, list[dict[str, str | int]]] = {}

    def add(self, user_id: str, role: str, content: str):
        entry = {
            "ts": int(time.time()),
            "role": role,
            "content": content,
        }
        self._store.setdefault(user_id, []).append(entry)

    def get(self, user_id: str, limit: int = 20):
        return self._store.get(user_id, [])[-limit:]

    def format_for_context(self, user_id: str, limit: int = 8):
        msgs = self.get(user_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

# global singleton for existing code paths in /ask and /memory
mem = MemoryStore()
