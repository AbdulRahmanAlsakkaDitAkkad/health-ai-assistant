# src/assistant/mental/phase8_dht.py
from __future__ import annotations

from datetime import datetime, date as date_cls
from typing import List, Optional, Dict, Any

from assistant.llm.ollama_client import OllamaClient
from assistant.memory.store import load_twin, save_twin, append_log
from assistant.schema.dht_models import (
    MoodLogEntry,
    EmotionAnalysisEntry,
    MentalHealthDailySummary,
    CoherenceRecommendationEntry,
    WearableRecord,
)

# ----------------------- Safety layer (non-clinical) -----------------------

# Simple keyword-based safety detection. This is intentionally conservative.
_URGENT_KEYWORDS = [
    "suicide", "kill myself", "end my life", "i want to die", "self harm", "self-harm",
    "hurt myself", "cut myself", "overdose", "can't go on", "no reason to live",
]
_CONCERNING_KEYWORDS = [
    "hopeless", "worthless", "panic", "can't sleep", "insomnia", "overwhelmed",
    "depressed", "anxious", "anxiety", "burnout", "exhausted", "can't cope",
]


def safety_flag_for_text(text: str) -> str:
    t = (text or "").lower()
    for k in _URGENT_KEYWORDS:
        if k in t:
            return "urgent"
    for k in _CONCERNING_KEYWORDS:
        if k in t:
            return "concerning"
    return "none"


def safety_message(flag: str) -> Optional[str]:
    if flag == "urgent":
        return (
            "I’m really sorry you’re feeling this way. I can’t help with anything harmful, "
            "but you deserve real, immediate support. If you’re in danger or might act on these thoughts, "
            "please contact your local emergency number now. If you can, reach out to a trusted person "
            "or a local crisis hotline in your country."
        )
    if flag == "concerning":
        return (
            "It sounds like this is weighing on you. I’m not a clinician, but talking to a trusted person "
            "or a licensed professional could really help—especially if these feelings persist or worsen."
        )
    return None


# ----------------------- Lightweight heuristics (fallback) -----------------------

_POS_WORDS = {
    "good", "great", "better", "calm", "okay", "fine", "relieved", "grateful", "happy",
    "content", "proud", "hope", "hopeful",
}
_NEG_WORDS = {
    "bad", "worse", "awful", "terrible", "anxious", "anxiety", "stressed", "stress", "panic",
    "sad", "tired", "exhausted", "overwhelmed", "angry", "upset", "hopeless",
}


def heuristic_sentiment(text: str) -> str:
    t = (text or "").lower()
    score = 0
    for w in _POS_WORDS:
        if w in t:
            score += 1
    for w in _NEG_WORDS:
        if w in t:
            score -= 1
    if score >= 2:
        return "positive"
    if score <= -2:
        return "negative"
    return "neutral"


def heuristic_categories(text: str) -> List[str]:
    t = (text or "").lower()
    cats: List[str] = []
    mapping = {
        "stress": ["stress", "stressed", "deadline", "pressure", "overwhelmed"],
        "anxiety": ["anxious", "anxiety", "panic", "worry", "nervous"],
        "fatigue": ["tired", "exhausted", "fatigue", "sleepy", "drained"],
        "calm": ["calm", "relaxed", "peaceful"],
        "sadness": ["sad", "down", "depressed", "hopeless"],
        "anger": ["angry", "mad", "irritated", "frustrated"],
    }
    for cat, keys in mapping.items():
        if any(k in t for k in keys):
            cats.append(cat)
    # keep stable order
    return list(dict.fromkeys(cats))[:5]


# ----------------------- LLM helpers (best-effort) -----------------------

def _try_llm_json(llm: Optional[OllamaClient], messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    if llm is None:
        return None
    try:
        raw = llm.chat(messages, temperature=0.1)
    except Exception:
        return None

    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = raw[start : end + 1]
    try:
        import json
        return json.loads(blob)
    except Exception:
        return None


# ----------------------- Phase 8 operations (DHT-backed) -----------------------

def add_mood_to_dht(
    user_id: str,
    mood_type: str,
    note: Optional[str] = None,
    emoji: Optional[str] = None,
) -> MoodLogEntry:
    twin = load_twin(user_id)
    entry = MoodLogEntry(mood_type=mood_type, note=note, emoji=emoji)
    twin.mood_logs.append(entry)
    save_twin(twin)
    append_log(
        user_id,
        response_type="mental_mood",
        content=f"{mood_type} {emoji or ''} {note or ''}".strip(),
        source="/add-mood",
    )
    return entry


def analyze_mood_text(
    user_id: str,
    text: str,
    llm: Optional[OllamaClient] = None,
) -> EmotionAnalysisEntry:
    flag = safety_flag_for_text(text)

    llm_payload = _try_llm_json(
        llm,
        [
            {"role": "system", "content": "You are a supportive, non-clinical assistant. Output ONLY JSON."},
            {
                "role": "user",
                "content": (
                    "Analyze the emotional tone of the text. Return JSON with keys: "
                    "sentiment (positive|neutral|negative), emotional_categories (array of strings), "
                    "suggested_reflection (string, 1-2 sentences). Text:\n"
                    f"{text}"
                ),
            },
        ],
    )

    if llm_payload:
        sentiment = str(llm_payload.get("sentiment", "neutral")).lower()
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"

        cats = llm_payload.get("emotional_categories") or []
        if isinstance(cats, str):
            cats = [c.strip() for c in cats.split(",") if c.strip()]
        if not isinstance(cats, list):
            cats = []
        cats = [str(c).strip() for c in cats if str(c).strip()][:6]

        reflection = str(llm_payload.get("suggested_reflection") or "").strip()
        if not reflection:
            reflection = (
                "Would you like to share what felt most difficult, and what (if anything) helped even a little?"
            )
        source = "llm"
    else:
        sentiment = heuristic_sentiment(text)
        cats = heuristic_categories(text)
        reflection = "Would you like to name one trigger today, and one small thing that could make tomorrow slightly easier?"
        source = "heuristic"

    entry = EmotionAnalysisEntry(
        text=text,
        sentiment=sentiment,  # type: ignore[arg-type]
        emotional_categories=cats,
        suggested_reflection=reflection,
        safety_flag=flag,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
    )

    twin = load_twin(user_id)
    twin.emotion_analysis.append(entry)
    save_twin(twin)
    append_log(
        user_id,
        response_type="mental_analysis",
        content=f"{sentiment} | {', '.join(cats)}",
        source="/analyze-mood",
    )
    return entry


def _parse_date(date_str: Optional[str]) -> date_cls:
    if not date_str:
        return datetime.utcnow().date()
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return datetime.utcnow().date()


def daily_mental_summary(
    user_id: str,
    date_str: Optional[str] = None,
    llm: Optional[OllamaClient] = None,
) -> MentalHealthDailySummary:
    target_date = _parse_date(date_str)

    twin = load_twin(user_id)
    moods = [m for m in twin.mood_logs if m.timestamp.date() == target_date]
    analyses = [a for a in twin.emotion_analysis if a.timestamp.date() == target_date]

    combined_text: List[str] = []
    for m in moods:
        bits = [m.mood_type]
        if m.note:
            bits.append(m.note)
        combined_text.append(" | ".join(bits))
    for a in analyses:
        combined_text.append(a.text)

    joined = "\n".join(combined_text).strip()

    flags = []
    if joined:
        flags.append(safety_flag_for_text(joined))
    flags.extend([a.safety_flag for a in analyses])

    flag = "none"
    if "urgent" in flags:
        flag = "urgent"
    elif "concerning" in flags:
        flag = "concerning"

    llm_payload = _try_llm_json(
        llm,
        [
            {"role": "system", "content": "You are a supportive, non-clinical assistant. Output ONLY JSON."},
            {
                "role": "user",
                "content": (
                    "Given the day's mood logs and notes, produce a short daily mental-health summary. "
                    "Return JSON with keys: summary (string), key_emotional_patterns (array), "
                    "stress_triggers (array), recommended_actions (array). Keep it non-clinical.\n\n"
                    f"DAY LOGS:\n{joined if joined else '(no logs)'}"
                ),
            },
        ],
    )

    if llm_payload and joined:
        summary = str(llm_payload.get("summary") or "").strip() or "Today had mixed moments."

        def _to_list(x: Any) -> List[str]:
            if isinstance(x, str):
                return [s.strip() for s in x.split("\n") if s.strip()]
            if isinstance(x, list):
                return [str(i).strip() for i in x if str(i).strip()]
            return []

        patterns_l = _to_list(llm_payload.get("key_emotional_patterns") or [])[:6]
        triggers_l = _to_list(llm_payload.get("stress_triggers") or [])[:6]
        actions_l = _to_list(llm_payload.get("recommended_actions") or [])[:6]
        source = "llm"
    else:
        mood_types = [m.mood_type for m in moods][-6:]
        cats: List[str] = []
        for a in analyses:
            cats.extend(a.emotional_categories)
        cats = list(dict.fromkeys([c for c in cats if c]))[:6]

        summary = "Logged moods: " + (", ".join(mood_types) if mood_types else "none") + "."
        patterns_l = cats or (["mixed"] if mood_types else [])

        triggers_l: List[str] = []
        t = joined.lower()
        for k in ["work", "deadline", "family", "money", "sleep", "health"]:
            if k in t:
                triggers_l.append(k)

        actions_l: List[str] = []
        if "fatigue" in cats or any(mt in ("tired", "exhausted") for mt in mood_types):
            actions_l.append("Try a consistent wind-down routine and aim for a slightly earlier bedtime.")
        if "stress" in cats or any(mt in ("stressed",) for mt in mood_types):
            actions_l.append("Try a 4-7-8 breathing exercise for 2 minutes.")
        if not actions_l:
            actions_l.append("Consider a short walk, hydration, and a brief check-in with someone you trust.")

        source = "heuristic"

    summary_obj = MentalHealthDailySummary(
        date=target_date.isoformat(),
        summary=summary,
        key_emotional_patterns=patterns_l,
        stress_triggers=triggers_l,
        recommended_actions=actions_l,
        safety_flag=flag,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
    )

    twin = load_twin(user_id)
    twin.mental_health_summaries.append(summary_obj)
    twin.mental_health_insights.append(summary_obj)  # manager-required storage label
    save_twin(twin)
    append_log(
        user_id,
        response_type="mental_daily_summary",
        content=summary,
        source="/daily-mental-summary",
    )
    return summary_obj


def recommend_relaxation(
    user_id: str,
    goal: Optional[str] = None,
    llm: Optional[OllamaClient] = None,
) -> CoherenceRecommendationEntry:
    twin = load_twin(user_id)
    last_mood = twin.mood_logs[-1] if twin.mood_logs else None
    last_analysis = twin.emotion_analysis[-1] if twin.emotion_analysis else None
    last_wearable: Optional[WearableRecord] = twin.wearables[-1] if twin.wearables else None

    mood_type = (last_mood.mood_type if last_mood else "").lower()
    sleep = last_wearable.sleep_hours if last_wearable else None
    cats = (last_analysis.emotional_categories if last_analysis else [])[:6]

    flag = "none"
    if last_analysis and last_analysis.safety_flag in ("concerning", "urgent"):
        flag = last_analysis.safety_flag
    if last_mood and last_mood.note:
        # max severity
        sev = {"none": 0, "concerning": 1, "urgent": 2}
        flag = max(flag, safety_flag_for_text(last_mood.note), key=lambda x: sev[x])

    llm_payload = _try_llm_json(
        llm,
        [
            {"role": "system", "content": "You are a supportive, non-clinical assistant. Output ONLY JSON."},
            {
                "role": "user",
                "content": (
                    "Recommend a short relaxation/coherence routine. Return JSON with keys: breathing_exercise, "
                    "audio_suggestion, grounding_exercise, light_routine, rationale. Keep it brief and non-clinical.\n\n"
                    f"goal={goal}\n"
                    f"mood_type={mood_type}\n"
                    f"sleep_hours={sleep}\n"
                    f"categories={cats}"
                ),
            },
        ],
    )

    if llm_payload:
        rec = CoherenceRecommendationEntry(
            goal=goal,
            breathing_exercise=str(llm_payload.get("breathing_exercise") or "").strip() or None,
            audio_suggestion=str(llm_payload.get("audio_suggestion") or "").strip() or None,
            grounding_exercise=str(llm_payload.get("grounding_exercise") or "").strip() or None,
            light_routine=str(llm_payload.get("light_routine") or "").strip() or None,
            rationale=str(llm_payload.get("rationale") or "").strip() or None,
            safety_flag=flag,  # type: ignore[arg-type]
            source="llm",
        )
    else:
        breathing = "4-7-8 breathing: inhale 4s, hold 7s, exhale 8s — repeat 4 cycles."
        grounding = "5-4-3-2-1 grounding: name 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste."
        audio = "Low-volume nature soundscape (rain/ocean) for 5–10 minutes."
        light = "Dim lights 30–60 minutes before sleep; avoid bright screens if possible."

        rationale_bits: List[str] = []
        if mood_type in ("stressed", "anxious") or "stress" in cats or "anxiety" in cats:
            rationale_bits.append("You logged stress/anxiety signals.")
        if sleep is not None and sleep < 6:
            rationale_bits.append("Sleep looks short, so a gentle wind-down may help.")
        if mood_type in ("tired", "exhausted") or "fatigue" in cats:
            rationale_bits.append("Fatigue signals suggest prioritizing rest and hydration.")

        rec = CoherenceRecommendationEntry(
            goal=goal,
            breathing_exercise=breathing,
            audio_suggestion=audio,
            grounding_exercise=grounding,
            light_routine=light if goal in (None, "sleep") or (sleep is not None and sleep < 6) else None,
            rationale=" ".join(rationale_bits) if rationale_bits else "Based on your recent mood and wearable signals.",
            safety_flag=flag,  # type: ignore[arg-type]
            source="heuristic",
        )

    twin.coherence_recommendations.append(rec)
    save_twin(twin)
    append_log(
        user_id,
        response_type="mental_recommendation",
        content=rec.rationale or "",
        source="/recommend-relaxation",
    )
    return rec
