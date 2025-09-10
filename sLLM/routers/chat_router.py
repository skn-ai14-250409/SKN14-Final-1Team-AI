from fastapi import APIRouter
from models.chat_model import ChatRequest
from services.langchain_service import chat_service

router = APIRouter()


@router.post("/chat")
async def chat(chat_request: ChatRequest):
    response, title = await chat_service.get_chat_response(chat_request)
    return {"response": response, "title": title}
