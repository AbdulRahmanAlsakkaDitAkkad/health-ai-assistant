# src/assistant/schema/dht_models.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class UserNote(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    text: str

class HealthHistoryItem(BaseModel):
    condition: str
    diagnosis_date: Optional[str] = None  # e.g., "2022-06"
    status: Optional[Literal["active", "resolved", "unknown"]] = "active"
    notes: Optional[str] = None

class SurgeryItem(BaseModel):
    name: str
    date: Optional[str] = None
    notes: Optional[str] = None

class FamilyHistoryItem(BaseModel):
    relation: str  # e.g., "mother", "father", "sibling", "grandparent"
    condition: str
    notes: Optional[str] = None

class DigitalHealthTwin(BaseModel):
    user_id: str
    # Free-form logs the user adds over time
    notes: List[UserNote] = Field(default_factory=list)
    # Medical history (diagnoses, surgeries)
    health_history: List[HealthHistoryItem] = Field(default_factory=list)
    surgeries: List[SurgeryItem] = Field(default_factory=list)
    # Family/genetic risk background
    family_history: List[FamilyHistoryItem] = Field(default_factory=list)
    # OCR / ingestion artifacts (simple list of strings for now)
    ingested_texts: List[str] = Field(default_factory=list)

    # A very small computed context we can pass to the LLM
    def render_context(self, max_chars: int = 2000) -> str:
        """Return a compact, human-readable context string for prompting."""
        parts: List[str] = []
        if self.health_history:
            hh = "; ".join(
                f"{h.condition}({h.status})"
                + (f"@{h.diagnosis_date}" if h.diagnosis_date else "")
                for h in self.health_history[:10]
            )
            parts.append(f"Health history: {hh}")
        if self.surgeries:
            su = "; ".join(
                f"{s.name}" + (f"@{s.date}" if s.date else "")
                for s in self.surgeries[:6]
            )
            parts.append(f"Surgeries: {su}")
        if self.family_history:
            fh = "; ".join(
                f"{f.relation}:{f.condition}"
                for f in self.family_history[:10]
            )
            parts.append(f"Family history: {fh}")
        if self.notes:
            last_notes = " | ".join(n.text.strip().replace("\n", " ") for n in self.notes[-5:])
            parts.append(f"Recent notes: {last_notes}")
        if self.ingested_texts:
            parts.append(f"Ingested items: {min(len(self.ingested_texts), 20)} docs")

        ctx = " | ".join(parts) if parts else "No significant prior data."
        return ctx[:max_chars]
