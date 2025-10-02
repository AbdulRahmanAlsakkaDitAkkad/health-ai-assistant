# file: src/assistant/ingest/file_ingestor.py

import os
from typing import Union
from assistant.ocr import parser
from assistant.memory.store import mem  # ✅ Add memory store

def read_file_bytes(file_path: str) -> bytes:
    """Read raw bytes from a local file path."""
    with open(file_path, "rb") as f:
        return f.read()

def detect_file_type(file_path: str) -> str:
    """Detect file type (pdf or image) from file extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".png", ".jpg", ".jpeg"]:
        return "image"
    else:
        raise ValueError("Unsupported file type")

def ingest_file(file: Union[str, bytes], file_type: str = None, user_id: str = "ocr_test") -> str:
    """
    Ingest a file from:
      - a file path (str), or
      - raw bytes (bytes) + file_type parameter.

    Stores extracted text in memory for the given user_id.
    Returns the extracted text.
    """
    # Determine input type and extract bytes
    if isinstance(file, str):
        file_type = detect_file_type(file)
        file_bytes = read_file_bytes(file)
        src_info = file
    elif isinstance(file, (bytes, bytearray)):
        if not file_type:
            raise ValueError("When passing raw bytes, you must provide file_type ('pdf' or 'image').")
        file_bytes = file
        src_info = "<uploaded-bytes>"
    else:
        raise ValueError("file must be a path (str) or raw bytes (bytes)")

    # OCR / PDF text extraction
    extracted_text = parser.extract_text(file_bytes, file_type=file_type)

    # Store in memory
    mem.add(user_id=user_id, role="user", content=extracted_text)

    print(f"✅ File Ingested ({file_type.upper()}): {src_info}")
    print(f"📥 Stored in memory for user: {user_id}")
    print("📝 Extracted Text:\n" + extracted_text)

    return extracted_text
