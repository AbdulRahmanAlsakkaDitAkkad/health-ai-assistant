# file: src/assistant/service/predictor.py
from datetime import datetime
from typing import List
from assistant.schema.dht_models import DigitalHealthTwin, Prediction

def predict_health_risks(twin: DigitalHealthTwin) -> List[Prediction]:
    predictions: List[Prediction] = []
    source_notes = []

    # Rule-based Diabetes Risk
    diabetes_risk = 0.1
    if any("diabetes" in h.lower() for h in twin.family_history):
        diabetes_risk += 0.3
        source_notes.append("family_history")
    if any("glucose" in t.lower() and "high" in t.lower() for t in twin.ingested_texts):
        diabetes_risk += 0.4
        source_notes.append("lab_report")
    if diabetes_risk > 0.1:
        predictions.append(Prediction(
            condition="diabetes",
            risk_score=min(diabetes_risk, 1.0),
            source="+".join(source_notes) or "unknown",
            timestamp=datetime.utcnow()
        ))

    # Rule-based Hypertension Risk
    hypertension_risk = 0.1
    if any("hypertension" in h.lower() for h in twin.health_history):
        hypertension_risk += 0.4
        source_notes.append("health_history")
    if any("blood pressure" in t.lower() and "high" in t.lower() for t in twin.ingested_texts):
        hypertension_risk += 0.3
        source_notes.append("lab_report")
    if hypertension_risk > 0.1:
        predictions.append(Prediction(
            condition="hypertension",
            risk_score=min(hypertension_risk, 1.0),
            source="+".join(source_notes) or "unknown",
            timestamp=datetime.utcnow()
        ))

    # Rule-based Sleep Risk
    sleep_risk = 0.1
    if any(w.sleep_hours and w.sleep_hours < 5 for w in twin.wearables):
        sleep_risk += 0.5
        source_notes.append("wearables")
    if sleep_risk > 0.1:
        predictions.append(Prediction(
            condition="sleep_quality",
            risk_score=min(sleep_risk, 1.0),
            source="+".join(source_notes) or "unknown",
            timestamp=datetime.utcnow()
        ))

    return predictions
