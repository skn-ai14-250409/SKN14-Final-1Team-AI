import logging
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool(parse_docstring=True)
def cto_blocker(question: str) -> str:
    """Reject questions related to CTO (Chief Technology Officer).

    Args:
        question: User's input text
    """
    return f"해당 질문은 CTO 관련된 내용이라 답변할 수 없습니다."


class LangChainChatService:
    def __init__(self):
        self.api_key = os.getenv("OLLAMA_API_URL")
        if not self.api_key:
            raise ValueError("OLLAMA_API_URL environment variable is required")

        self.model_name = os.getenv("OLLAMA_MODEL")

        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.api_key,
            api_key="ollama",
            temperature=0.2,
        )

        # Bind tools (계산기 포함)
        self.llm_tools = self.llm.bind_tools([cto_blocker])

    async def get_chat_response(self, history: list[dict]) -> str:
        try:
            system_message = """
            당신은 사내 지식을 활용하여 사용자의 질문에 정확하고 유용한 답변을 제공하는 AI 비서입니다.
            다음 지침을 따르세요:
            1. 항상 정중하고 전문적인 어조를 유지하세요.
            2. 사용자의 질문을 주의 깊게 읽고 이해한 후 답변
            3. 답변이 불확실한 경우, "잘 모르겠습니다"라고 솔직하게 말하세요.
            4. 정보를 활용할 때는 신뢰할 수 있는 출처를 사용하세요.
            5. 답변이 너무 길어지지 않도록 주의하세요.
            6. 사용자가 추가 질문을 할 수 있도록 격려하세요.

            사용자가 CTO 관련 질문을 하면 반드시 "cto_blocker" 툴을 호출하세요.
            """

            # 대화 기록 변환
            state = [SystemMessage(content=system_message)]
            for h in history:
                if h["type"] == "user":
                    state.append(HumanMessage(content=h["content"]))
                elif h["type"] == "assistant":
                    state.append(AIMessage(content=h["content"]))

            # LLM 호출
            ai = self.llm_tools.invoke(state)
            logger.info(f"LLM Raw Response: {ai}")
            state.append(ai)

            # 툴 호출 확인
            tool_calls = getattr(ai, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    if call["name"] == "cto_blocker":
                        question = call["args"]["question"]
                        result = cto_blocker.invoke({"question": question})
                        state.append(
                            ToolMessage(tool_call_id=call["id"], content=result)
                        )
                # 툴 실행 결과 반영 후 다시 LLM 호출
                ai = self.llm_tools.invoke(state)
                state.append(ai)

            # 최종 답변 추출
            assistant_reply = state[-1].content

            # <think>...</think> 태그 제거 (열림 없음 + 닫힘만 있어도 제거)
            assistant_reply = re.sub(
                r"<think>.*?</think>", "", assistant_reply, flags=re.S
            )
            assistant_reply = assistant_reply.replace("</think>", "")
            assistant_reply = assistant_reply.strip()

            return assistant_reply
        except Exception as e:
            raise Exception(f"Failed to get response from AI: {str(e)}")


# Create global service instance
chat_service = LangChainChatService()
