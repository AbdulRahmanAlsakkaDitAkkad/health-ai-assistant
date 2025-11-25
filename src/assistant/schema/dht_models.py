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


# ---------- Wearable data (for predictive modeling) ----------
class WearableRecord(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    steps: Optional[int] = None
    heart_rate: Optional[float] = None
    sleep_hours: Optional[float] = None


# ---------- Prediction results ----------
class RiskPrediction(BaseModel):
    condition: str
    risk_score: float  # between 0 and 1
    explanation: str


# ---------- Phase 7: Medications, Appointments, Notes ----------
class MedicationItem(BaseModel):
    name: str
    dosage: Optional[str] = None      # e.g. "500 mg"
    schedule: Optional[str] = None    # e.g. "twice daily", "at night"
    start_date: Optional[str] = None  # e.g. "2025-10-01"
    end_date: Optional[str] = None    # e.g. "2025-11-01"
    notes: Optional[str] = None


class AppointmentItem(BaseModel):
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DoctorNote(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_text: str
    summary: Optional[str] = None
    visit_type: Optional[Literal["doctor", "therapist", "other"]] = None


# ---------- Logging / explainability ----------
class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_type: Literal[
        "summary",            # LLM summaries
        "ask_answer",         # answers from /ask
        "followup_questions", # output from /suggest-followup-questions
        "prediction",         # Phase 5 risk / forecast
        "qr",                 # Phase 6 QR generation / decoding
        "medication",         # Phase 7 medication changes
        "appointment",        # Phase 7 appointment actions
        "note_summary",       # Phase 7 doctor/therapist note summaries
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

    # Phase 5 extensions
    wearables: List[WearableRecord] = Field(default_factory=list)
    predictions: List[RiskPrediction] = Field(default_factory=list)

    # Phase 7 extensions
    medications: List[MedicationItem] = Field(default_factory=list)
    appointments: List[AppointmentItem] = Field(default_factory=list)
    doctor_notes: List[DoctorNote] = Field(default_factory=list)

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

        if self.medications:
            meds = "; ".join(m.name for m in self.medications[-5:])
            parts.append(f"Medications: {meds}")

        if self.appointments:
            # Just show the latest appointment title + time
            latest_appt = sorted(self.appointments, key=lambda a: a.start_time)[-1]
            parts.append(f"Last appointment: {latest_appt.title} @ {latest_appt.start_time.isoformat()}")

        if self.notes:
            last_notes = " | ".join(n.text.strip().replace("\n", " ") for n in self.notes[-5:])
            parts.append(f"Recent notes: {last_notes}")

        if self.ingested_items:
            parts.append(f"Ingested docs: {min(len(self.ingested_items), 20)}")

        if self.wearables:
            parts.append(f"Wearable records: {len(self.wearables)}")

        if self.predictions:
            preds = "; ".join(f"{p.condition}:{p.risk_score:.2f}" for p in self.predictions[-5:])
            parts.append(f"Predictions: {preds}")

        ctx = " | ".join(parts) if parts else "No significant prior data."
        return ctx[:max_chars]
