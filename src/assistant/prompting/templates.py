# src/assistant/prompting/templates.py

FOLLOWUP_SYSTEM_PROMPT = """You are a helpful health assistant.
Given the patient's context and recent findings, generate clear, concise follow-up
questions the patient could ask a healthcare professional. Avoid diagnosis.
Return a numbered list of 5-10 questions. Keep each question short.
"""

def build_personalized_user_prompt(user_question: str, context: str) -> str:
    return f"""Patient context:
{context}

User question:
{user_question}

Instructions:
- Use the patient context to personalize your answer.
- Be clear, non-alarming, and avoid giving medical diagnosis.
- If information is insufficient, state what is missing and suggest next steps.
"""

def build_followup_prompt(lab_or_summary_text: str, context: str) -> str:
    return f"""Patient context:
{context}

Recent summary/lab info:
{lab_or_summary_text}

Task:
List the top 5-10 follow-up questions the patient can ask their doctor.
Be specific to the context and findings. No diagnosis."""
