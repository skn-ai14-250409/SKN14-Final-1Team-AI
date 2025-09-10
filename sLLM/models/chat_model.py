from typing import Dict, List, Literal
from pydantic import BaseModel


class ChatRequest(BaseModel):
    history: List[Dict]
    permission: Literal["cto", "backend", "frontend", "data-ai", "none"] = "none"
    tone: Literal["formal", "informal"] = "formal"
