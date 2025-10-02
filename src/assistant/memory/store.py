from __future__ import annotations
from typing import List, Dict, Any
from collections import defaultdict, deque
import time


class MemoryStore:
    """
    Very simple in-memory conversation store.
    - Keeps the last N messages per user (bounded by max_per_user).
    - Meant for prototyping; later we can swap to SQLite/Postgres.
    """
    def __init__(self, max_per_user: int = 50):
        self._store = defaultdict(lambda: deque(maxlen=max_per_user))

    def add(self, user_id: str, role: str, content: str) -> None:
        self._store[user_id].append({
            "ts": time.time(),
            "role": role,
            "content": content
        })

    def get(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        items = list(self._store[user_id])
        if limit <= 0:
            return items
        return items[-limit:]

    def format_for_context(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Return last messages formatted like LLM expects: [{role, content}, ...]."""
        return [{"role": m["role"], "content": m["content"]} for m in self.get(user_id, limit)]


# Global singleton for simplicity (FastAPI process-local).
mem = MemoryStore(max_per_user=100)
