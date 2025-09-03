from fastapi import APIRouter
from models.chat_model import ChatMessage

router = APIRouter()


@router.post("/chat")
def chat(item: ChatMessage):
    return {"response": "test"}
