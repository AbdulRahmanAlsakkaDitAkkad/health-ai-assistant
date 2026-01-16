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
    status: Optional[str] = "active"      # active, resolved, etc
    notes: Optional[str] = None


class SurgeryItem(BaseModel):
    name: str
    date: Optional[str] = None            # e.g. "2021-06-10"
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


# ---------- Phase 8: Mental Health & Coherence Intelligence ----------
class MoodLogEntry(BaseModel):
    """User-submitted mood entry (non-clinical)."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mood_type: str  # e.g. "happy" | "stressed" | "anxious" | "neutral" | "tired"
    note: Optional[str] = None
    emoji: Optional[str] = None


class EmotionAnalysisEntry(BaseModel):
    """Structured interpretation of a mood/note (LLM or heuristic)."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    text: str
    sentiment: Literal["positive", "neutral", "negative"]
    emotional_categories: List[str] = Field(default_factory=list)
    suggested_reflection: str
    safety_flag: Literal["none", "concerning", "urgent"] = "none"
    source: Literal["llm", "heuristic"] = "heuristic"


class MentalHealthDailySummary(BaseModel):
    """Per-day mental health summary (non-diagnostic)."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    date: str  # "YYYY-MM-DD"
    summary: str
    key_emotional_patterns: List[str] = Field(default_factory=list)
    stress_triggers: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    safety_flag: Literal["none", "concerning", "urgent"] = "none"
    source: Literal["llm", "heuristic"] = "heuristic"


class CoherenceRecommendationEntry(BaseModel):
    """Relaxation & coherence recommendations that adapt to mood + wearable signals."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    goal: Optional[str] = None  # e.g. "calm" | "sleep" | "focus"
    breathing_exercise: Optional[str] = None
    audio_suggestion: Optional[str] = None
    grounding_exercise: Optional[str] = None
    light_routine: Optional[str] = None
    rationale: Optional[str] = None
    safety_flag: Literal["none", "concerning", "urgent"] = "none"
    source: Literal["llm", "heuristic"] = "heuristic"


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
        # Phase 8
        "mental_mood",          # mood log
        "mental_analysis",      # sentiment/emotion analysis
        "mental_daily_summary", # daily mental summary
        "mental_recommendation" # relaxation/coherence recommendations
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

    # Phase 8 extensions
    mood_logs: List[MoodLogEntry] = Field(default_factory=list)
    emotion_analysis: List[EmotionAnalysisEntry] = Field(default_factory=list)
    mental_health_summaries: List[MentalHealthDailySummary] = Field(default_factory=list)
    # Alias section requested by roadmap (stored redundantly for compatibility)
    mental_health_insights: List[MentalHealthDailySummary] = Field(default_factory=list)
    coherence_recommendations: List[CoherenceRecommendationEntry] = Field(default_factory=list)
    
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
  
        if self.family_history:
            fh = "; ".join(f"{f.relation}:{f.condition}" for f in self.family_history[-10:])
            parts.append(f"Family: {fh}")
            
        if self.surgeries:
            sx = "; ".join(s.name for s in self.surgeries[-10:])
            parts.append(f"Surgeries: {sx}")
            
        if self.medications:
            meds = "; ".join(m.name for m in self.medications[-10:])
            parts.append(f"Meds: {meds}")

        if self.appointments:
            parts.append(f"Appointments: {min(len(self.appointments), 20)}")

        if self.doctor_notes:
            parts.append(f"Doctor notes: {min(len(self.doctor_notes), 20)}")

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

        # Phase 8 mental context (keep compact)
        if self.mood_logs:
            last = self.mood_logs[-1]
            mood_bits = [last.mood_type]
            if last.emoji:
                mood_bits.append(last.emoji)
            if last.note:
                mood_bits.append(last.note.strip().replace("\n", " ")[:140])
            parts.append("Latest mood: " + " ".join(mood_bits))

        # Prefer the most recent daily summary/insight if present
        latest_summaries: List[MentalHealthDailySummary] = []
        if self.mental_health_insights:
            latest_summaries = self.mental_health_insights
        elif self.mental_health_summaries:
            latest_summaries = self.mental_health_summaries

        if latest_summaries:
            last_sum = latest_summaries[-1]
            parts.append("Mental summary: " + last_sum.summary.strip().replace("\n", " ")[:220])

        ctx = " | ".join(parts) if parts else "No significant prior data."
        return ctx[:max_chars]
