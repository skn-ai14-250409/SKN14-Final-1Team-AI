
from .service import chat_service


def chat(chat_request):
    response, result = chat_service.get_chat_response(chat_request)
    return {"response": response, "result": result}