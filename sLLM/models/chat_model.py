from typing import Dict, List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    history: List[Dict]
