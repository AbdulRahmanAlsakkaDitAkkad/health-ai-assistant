# file: src/assistant/llm/ollama_client.py
from __future__ import annotations

import json
import requests
from typing import List, Dict, Any

DEFAULT_OLLAMA_URL = "http://localhost:11434"

class OllamaClient:
    """
    Wrapper for the local Ollama REST API.

    - Tries POST /api/chat first (messages format).
    - Falls back to POST /api/generate (single prompt).
    - Has extra helpers for Digital Twin context and follow-up questions.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = "mistral:latest",
        timeout_s: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    # ---------------------------------------
    # Core chat method (your original design)
    # ---------------------------------------
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        url_chat = f"{self.base_url}/api/chat"
        payload_chat: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 256},
        }

        try:
            r = requests.post(url_chat, json=payload_chat, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception as e:
            print("WARN: /api/chat failed, falling back:", repr(e))

        # --- fallback to /api/generate ---
        url_gen = f"{self.base_url}/api/generate"
        prompt = "\n".join(m["content"] for m in messages if m.get("role") in ("system", "user"))
        payload_gen: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 256},
        }

        r2 = requests.post(url_gen, json=payload_gen, timeout=self.timeout_s)
        r2.raise_for_status()
        data2 = r2.json()
        resp = data2.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError(f"Ollama returned no content. Body: {r2.text[:200]}")
        return resp.strip()

    # ---------------------------------------
    # NEW: simple generate helper
    # ---------------------------------------
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()

    # ---------------------------------------
    # NEW: Digital Twin helpers
    # ---------------------------------------
    def answer_with_context(self, user_prompt: str, context: str) -> str:
        """
        Answer user question using their Digital Twin context.
        """
        from assistant.prompting.templates import build_personalized_user_prompt
        prompt = build_personalized_user_prompt(user_prompt, context)
        return self.generate(prompt)

    def generate_followup_questions(self, summary_or_text: str, context: str) -> str:
        """
        Suggest follow-up questions for a doctor visit.
        """
        from assistant.prompting.templates import FOLLOWUP_SYSTEM_PROMPT, build_followup_prompt
        up = build_followup_prompt(summary_or_text, context)
        return self.chat(
            [
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": up},
            ]
        )
