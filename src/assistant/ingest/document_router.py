# file: src/assistant/ingest/document_router.py

from typing import Literal, Dict, List
from assistant.llm.ollama_client import OllamaClient 
import os 
import warnings

DocumentType = Literal["lab_report", "prescription", "general_text"]

# Define question sets for structured extraction (Phase 3)
LAB_REPORT_QUESTIONS = [
    "What is the Patient Name?",
    "What is the Date of Birth?",
    "What is the Test Ordered?",
    "What are the Results?",
    "What is the Impression?",
    "What are the Recommendations?"
]

PRESCRIPTION_QUESTIONS = [
    "What is the Patient Name?",
    "What is the Drug Name?",
    "What is the Dosage?",
    "What are the Directions (SIG)?",
    "How many refills are authorized?",
]

# Simple Heuristics for quick classification
CLASSIFICATION_KEYWORDS: Dict[DocumentType, List[str]] = {
    "lab_report": ["hemoglobin", "wbc count", "platelets", "normal range", "g/dl"],
    "prescription": ["rx", "dispense", "dosage", "sig", "refills", "drug name"]
}

def classify_document(text: str) -> DocumentType:
    """Classifies document type using keyword heuristics."""
    text_lower = text.lower()

    # 1. Check for Lab Report keywords
    if any(keyword in text_lower for keyword in CLASSIFICATION_KEYWORDS["lab_report"]):
        return "lab_report"

    # 2. Check for Prescription keywords
    if any(keyword in text_lower for keyword in CLASSIFICATION_KEYWORDS["prescription"]):
        return "prescription"

    # 3. Default fallback
    return "general_text"

def get_question_set(doc_type: DocumentType) -> List[str]:
    """Returns the appropriate question set for the Donut model."""
    if doc_type == "lab_report":
        return LAB_REPORT_QUESTIONS
    if doc_type == "prescription":
        return PRESCRIPTION_QUESTIONS
    return []