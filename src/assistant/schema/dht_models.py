# src/assistant/schema/dht_models.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ---------- Core data units ----------
class UserNote(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    text: str


class HealthHistoryItem(BaseModel):
    condition: str
    diagnosis_date: Optional[str] = None  # e.g. "2022-05-01" or "2022-05"
    status: Optional[Literal["active", "resolved", "unknown"]] = "active"
    notes: Optional[str] = None


class SurgeryItem(BaseModel):
    name: str
    date: Optional[str] = None
    notes: Optional[str] = None


class FamilyHistoryItem(BaseModel):
    relation: str  # e.g. mother, father, sibling
    condition: str
    notes: Optional[str] = None


class IngestedItem(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str  # filename or "user_upload" etc
    text: str


# ---------- Logging / explainability ----------
class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_type: Literal[
        "summary",           # LLM summaries
        "ask_answer",        # answers from /ask
        "followup_questions" # output from /suggest-followup-questions
    ]
    content: str
    source: Optional[str] = None  # e.g., "ingest:report.pdf", "ask:<question>"


# ---------- The Digital Health Twin ----------
class DigitalHealthTwin(BaseModel):
    user_id: str

    notes: List[UserNote] = Field(default_factory=list)
    health_history: List[HealthHistoryItem] = Field(default_factory=list)
    surgeries: List[SurgeryItem] = Field(default_factory=list)
    family_history: List[FamilyHistoryItem] = Field(default_factory=list)
    ingested_items: List[IngestedItem] = Field(default_factory=list)

    # Explainability / trace
    logs: List[LogEntry] = Field(default_factory=list)

    # Compact context for prompting
    def render_context(self, max_chars: int = 2000) -> str:
        parts: List[str] = []
        if self.health_history:
            hh = "; ".join(
                f"{h.condition}({h.status})" + (f"@{h.diagnosis_date}" if h.diagnosis_date else "")
                for h in self.health_history[-10:]
            )
            parts.append(f"Health history: {hh}")
        if self.surgeries:
            su = "; ".join(
                f"{s.name}" + (f"@{s.date}" if s.date else "")
                for s in self.surgeries[-6:]
            )
            parts.append(f"Surgeries: {su}")
        if self.family_history:
            fh = "; ".join(
                f"{f.relation}:{f.condition}"
                for f in self.family_history[-10:]
            )
            parts.append(f"Family history: {fh}")
        if self.notes:
            last_notes = " | ".join(n.text.strip().replace("\n", " ") for n in self.notes[-5:])
            parts.append(f"Recent notes: {last_notes}")
        if self.ingested_items:
            parts.append(f"Ingested docs: {min(len(self.ingested_items), 20)}")

        ctx = " | ".join(parts) if parts else "No significant prior data."
        return ctx[:max_chars]
