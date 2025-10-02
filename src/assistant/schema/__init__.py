from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AskReq(BaseModel):
    question: str = Field(..., description="User question or instruction")
    user_id: Optional[str] = Field(
        default="demo",
        description="User identifier for memory/personalization. Defaults to 'demo'."
    )


class AskRes(BaseModel):
    answer: str


class MemoryItem(BaseModel):
    ts: float
    role: str
    content: str


class MemoryList(BaseModel):
    user_id: str
    items: List[MemoryItem]


class Health(BaseModel):
    status: str = "ok"
