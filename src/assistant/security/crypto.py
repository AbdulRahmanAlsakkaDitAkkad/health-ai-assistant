# src/assistant/security/crypto.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

KEY_ENV = "DHT_QR_SECRET"
KEY_PATH = Path(__file__).resolve().parents[3] / "data" / "keys" / "fernet.key"

def _ensure_key() -> bytes:
    # 1) Try env var first
    env_key = os.getenv(KEY_ENV)
    if env_key:
        try:
            return env_key.encode("utf-8")
        except Exception:
            pass

    # 2) Try file
    if KEY_PATH.exists():
        k = KEY_PATH.read_bytes()
        if k:
            return k

    # 3) Generate new key and persist to file
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    k = Fernet.generate_key()
    KEY_PATH.write_bytes(k)
    return k

def get_fernet() -> Fernet:
    key = _ensure_key()
    return Fernet(key)

def encrypt_bytes(data: bytes) -> bytes:
    return get_fernet().encrypt(data)

def decrypt_bytes(token: bytes) -> bytes:
    return get_fernet().decrypt(token)
