# src/assistant/memory/digital_twin.py
from __future__ import annotations
from typing import Optional, List
from assistant.schema.dht_models import (
    DigitalHealthTwin,
    UserNote,
    HealthHistoryItem,
    SurgeryItem,
    FamilyHistoryItem,
)
from assistant.memory.store import load_twin, save_twin

class DigitalTwinManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.twin: DigitalHealthTwin = load_twin(user_id)

    # ---- Adders ----
    def add_user_note(self, text: str) -> DigitalHealthTwin:
        self.twin.notes.append(UserNote(text=text))
        save_twin(self.twin)
        return self.twin

    def add_health_history(self, items: List[HealthHistoryItem]) -> DigitalHealthTwin:
        self.twin.health_history.extend(items)
        save_twin(self.twin)
        return self.twin

    def add_surgeries(self, items: List[SurgeryItem]) -> DigitalHealthTwin:
        self.twin.surgeries.extend(items)
        save_twin(self.twin)
        return self.twin

    def add_family_history(self, items: List[FamilyHistoryItem]) -> DigitalHealthTwin:
        self.twin.family_history.extend(items)
        save_twin(self.twin)
        return self.twin

    # ---- Context for LLM ----
    def build_context(self) -> str:
        return self.twin.render_context()

    # ---- Accessor ----
    def snapshot(self) -> DigitalHealthTwin:
        return self.twin
