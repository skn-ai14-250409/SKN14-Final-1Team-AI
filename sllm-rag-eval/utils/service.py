import json
import logging
import re
import gdown
import os, shutil, tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 기존 벡터 저장소 로드
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
)


@tool(parse_docstring=True)
def cto_search(question: str) -> str:
    """CTO 관련 질문일 경우, 모든 team(cto, backend, frontend, data_ai)을 함께 검색합니다.

    Args:
        question: 사용자가 입력한 질문
    """
    outputs = []
    results = vectorstore.similarity_search(question, k=10)
    if results:
        for i, result in enumerate(results, start=1):
            outputs.append(
                f"Result {i}:\n"
                f"Content: {result.page_content}\n"
                f"Metadata: {result.metadata}"
            )
    return "\n\n".join(outputs) if outputs else "CTO 관련된 답변을 찾지 못했습니다."



# 툴 정의
@tool(parse_docstring=True)
def team_search(question: str, team: str) -> str:
    """사내 벡터DB에서 특정 team 관련 문서를 검색합니다.

    Args:
        question: 사용자가 입력한 질문
        team: 검색할 팀/직급 (예: 'frontend', 'backend', 'data_ai')
    """
    try:
        # team 기반 검색 수행
        results = vectorstore.similarity_search(question, k=10, filter={"role": team})
        if not results:
            return f"{team} 관련된 답변을 찾지 못했습니다."

        output = []
        for i, result in enumerate(results, start=1):
            output.append(
                f"[{team}] Result {i}:\n"
                f"Content: {result.page_content}\n"
                f"Metadata: {result.metadata}"
            )
        return "\n\n".join(output)
    except Exception as e:
        return f"{team} 검색 중 오류 발생: {e}"


# 서비스 클래스
class LangChainChatService:
    def __init__(self):
        self.api_url = os.getenv("VLLM_API_URL")
        if not self.api_url:
            raise ValueError("VLLM_API_URL environment variable is required")

        self.model_name = os.getenv("VLLM_MODEL")
        self.api_key = os.getenv("VLLM_API_KEY")

        # OpenAI 호환 LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_base=self.api_url,
            openai_api_key=self.api_key,
            temperature=0.2,
        )

    def get_chat_response(self, request) -> str:
        history = request['history']
        permission = request['permission']
        tone = request['tone']
        result = '없음'  # result 변수 초기화
        try:
            # 톤별 가이드
            if tone == "formal":
                tone_instruction = (
                    "기존의 말투는 잊고 정중하고 사무적인 어조로 답변하세요."
                )
            elif tone == "informal":
                tone_instruction = "기존의 말투는 잊고 가볍고 친근한 반말로 답변하세요."

            # permission별 툴 선택
            if permission == "cto":
                # 툴 바인딩
                llm_tools = self.llm.bind_tools([cto_search])
                tool_prompt = f"사용자는 CTO며 반드시 cto_search 툴을 호출하세요."
            elif permission in ["backend", "frontend", "data_ai"]:
                # 툴 바인딩
                llm_tools = self.llm.bind_tools([team_search])
                tool_prompt = f"사용자는 {permission}팀이며 반드시 team={permission}으로 team_search 툴을 호출하세요."
            else:
                llm_tools = self.llm.bind_tools([])
                tool_prompt = f""

            # 시스템 메시지 생성
            system_message = f"""
            당신은 사내 지식을 활용하여 사용자의 질문에 정확하고 유용한 답변을 제공하는 한국인 AI 비서입니다.
            다음 지침을 따르세요:
            1. {tone_instruction}
            2. 사실에 기반한 정보를 사용하세요.
            3. 답변이 불확실한 경우, "잘 모르겠습니다"라고 솔직하게 말하세요.
            4. 답변이 너무 길지 않게 하세요.
            5. {tool_prompt}
            """

            # 대화 기록 변환
            state = [SystemMessage(content=system_message)]
            for h in history:
                if h["role"] == "user":
                    state.append(HumanMessage(content=h["content"]))
                elif h["role"] == "assistant":
                    state.append(AIMessage(content=h["content"]))

            # LLM한테 Tool Call 호출을 확인한다.
            """
            <think>...</think>
            <tool_call>...</tool_call>
            형식을 리턴한다.
            """

            llm_tool = llm_tools.invoke(state)
            state.append(llm_tool)

            # <tool_call> 분석
            assistant_reply = state[-1].content
            matches = re.findall(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", assistant_reply, flags=re.S
            )

            extra_calls = []
            for m in matches:
                try:
                    extra_calls.append(json.loads(m))
                except json.JSONDecodeError:
                    pass

            if extra_calls:
                for call in extra_calls:
                    if call["name"] == "team_search":
                        result = team_search.invoke(call["arguments"])
                        state.append(
                            ToolMessage(
                                tool_call_id=call.get("id", "extra1"), content=result
                            )
                        )
                    elif call["name"] == "cto_search":
                        result = cto_search.invoke(call["arguments"])
                        state.append(
                            ToolMessage(
                                tool_call_id=call.get("id", "extra2"), content=result
                            )
                        )

                # 툴 결과 반영 후 재호출
                llm_res = llm_tools.invoke(state)
                state.append(llm_res)

            # 최종 답변 정리
            assistant_reply = state[-1].content
            assistant_reply = re.sub(
                r"<think>.*?</think>", "", assistant_reply, flags=re.S
            )
            assistant_reply = assistant_reply.replace("</think>", "").strip()

            return assistant_reply, result

        except Exception as e:
            raise Exception(f"Failed to get response from AI: {str(e)}")


chat_service = LangChainChatService()