# src/assistant/memory/digital_twin.py
from __future__ import annotations
from typing import List, Optional

from assistant.schema.dht_models import (
    DigitalHealthTwin,
    UserNote,
    HealthHistoryItem,
    FamilyHistoryItem,
    SurgeryItem,
    WearableRecord,     # ✅ correct name
    RiskPrediction,     # ✅ fixed name
)
from assistant.memory.store import (
    load_twin,
    save_twin,
    append_log,
    dedupe_health_history,
    dedupe_family_history,
    dedupe_surgeries,
    dedupe_notes,
)

class DigitalTwinManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.twin: DigitalHealthTwin = load_twin(user_id)

    # ---- Adders with validation/dedupe ----
    def add_user_note(self, text: str) -> DigitalHealthTwin:
        self.twin.notes = dedupe_notes(self.twin.notes, [UserNote(text=text)])
        save_twin(self.twin)
        return self.twin

    def add_health_history(self, items: List[HealthHistoryItem]) -> DigitalHealthTwin:
        self.twin.health_history = dedupe_health_history(self.twin.health_history, items)
        save_twin(self.twin)
        return self.twin

    def add_surgeries(self, items: List[SurgeryItem]) -> DigitalHealthTwin:
        self.twin.surgeries = dedupe_surgeries(self.twin.surgeries, items)
        save_twin(self.twin)
        return self.twin

    def add_family_history(self, items: List[FamilyHistoryItem]) -> DigitalHealthTwin:
        self.twin.family_history = dedupe_family_history(self.twin.family_history, items)
        save_twin(self.twin)
        return self.twin

    # ---- New: Wearable data ingestion ----
    def add_wearables(self, items: List[WearableRecord]) -> DigitalHealthTwin:
        self.twin.wearables.extend(items)
        save_twin(self.twin)
        return self.twin

    # ---- New: Prediction results ----
    def add_predictions(self, items: List[RiskPrediction]) -> DigitalHealthTwin:
        self.twin.predictions.extend(items)
        save_twin(self.twin)
        return self.twin

    # ---- Logging / Explainability ----
    def log_event(self, response_type: str, content: str, source: Optional[str] = None) -> None:
        append_log(self.user_id, response_type=response_type, content=content, source=source)
        # refresh local copy
        self.twin = load_twin(self.user_id)

    # ---- Context for LLM ----
    def build_context(self) -> str:
        return self.twin.render_context()

    # ---- Snapshot ----
    def snapshot(self) -> DigitalHealthTwin:
        return load_twin(self.user_id)
