from __future__ import annotations

import json
from typing import List, Dict, Any

import requests


class OllamaClient:
    """
    Robust wrapper for the local Ollama REST API.

    - Tries POST /api/chat first (messages format).
    - Falls back to POST /api/generate (single prompt).
    - Uses a higher timeout to tolerate initial model load on CPU.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral:latest",
        timeout_s: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """
        Send a list of messages and return the assistant's text reply.

        messages example:
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Summarize ..."}
        ]
        """
        # --- 1) Try /api/chat with messages ---
        url_chat = f"{self.base_url}/api/chat"
        payload_chat: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # single JSON response (per Ollama REST API)
            "options": {"temperature": temperature, "num_predict": 128},
        }

        try:
            r = requests.post(url_chat, json=payload_chat, timeout=self.timeout_s)
            print("DEBUG /api/chat status:", r.status_code)
            print("DEBUG /api/chat body:", r.text[:600])
            r.raise_for_status()

            data = r.json()
            # Expected shape: {"message":{"role":"assistant","content":"..."}}
            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

            print("WARN: /api/chat had no message.content; falling back to /api/generate.")

        except Exception as e:
            print("WARN: /api/chat failed:", repr(e))

        # --- 2) Fallback to /api/generate with a single prompt ---
        url_gen = f"{self.base_url}/api/generate"
        prompt = "\n".join(m["content"] for m in messages if m.get("role") in ("system", "user"))
        payload_gen: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # single JSON response (per Ollama REST API)
            "options": {"temperature": temperature, "num_predict": 128},
        }

        r2 = requests.post(url_gen, json=payload_gen, timeout=self.timeout_s)
        print("DEBUG /api/generate status:", r2.status_code)
        print("DEBUG /api/generate body:", r2.text[:600])
        r2.raise_for_status()

        data2 = r2.json()
        # Expected shape: {"response":"..."}
        resp = data2.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError(f"Ollama returned no content. Body: {r2.text[:600]}")
        return resp.strip()
