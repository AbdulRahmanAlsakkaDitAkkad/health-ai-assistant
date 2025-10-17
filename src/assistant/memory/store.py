# file: src/assistant/memory/store.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from assistant.schema.dht_models import DigitalHealthTwin

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
            # if file corrupted, return fresh twin
            return DigitalHealthTwin(user_id=user_id)
    return DigitalHealthTwin(user_id=user_id)

def save_twin(twin: DigitalHealthTwin) -> None:
    p = _user_path(twin.user_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(twin.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)

def append_ingested_text(user_id: str, text: str) -> None:
    twin = load_twin(user_id)
    twin.ingested_texts.append(text)
    save_twin(twin)

# -------------------------------------------------------------------
# Legacy in-memory store (for backward compatibility with api.py)
# -------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        self._store: dict[str, list[dict[str, str | int]]] = {}

    def add(self, user_id: str, role: str, content: str):
        """
        Add a memory entry with a timestamp, role, and content.
        """
        entry = {
            "ts": int(time.time()),   # timestamp in seconds
            "role": role,
            "content": content,
        }
        self._store.setdefault(user_id, []).append(entry)

    def get(self, user_id: str, limit: int = 20):
        """
        Get the most recent `limit` entries for this user.
        """
        return self._store.get(user_id, [])[-limit:]

    def format_for_context(self, user_id: str, limit: int = 8):
        """
        Return messages formatted for LLM context.
        """
        msgs = self.get(user_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

# global singleton for existing code
mem = MemoryStore()
