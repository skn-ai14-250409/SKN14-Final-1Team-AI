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

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "chroma_db")
DRIVE_URL = "https://drive.google.com/drive/folders/1STUOUcZWZatvaK54_B9qxv0mEEjYAub0"


# 구글 드라이브 링크 안에 있는 파일을 통으로 가져와서 chroma_db 폴더에 넣기
def download_drive_folder_to_chroma_db(folder_url: str, target_dir: Path):
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        gdown.download_folder(url=folder_url, output=td, quiet=False, use_cookies=False)
        entries = [Path(td) / name for name in os.listdir(td)]
        src_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(td)

        for p in src_root.iterdir():
            dst = target_dir / p.name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(p), str(dst))

    if not (target_dir / "chroma.sqlite3").exists():
        raise RuntimeError(f"'chroma.sqlite3'가 없습니다: {target_dir}")


def create_chroma_db():
    HERE = Path(__file__).resolve().parent
    FOLDER_URL = DRIVE_URL
    download_drive_folder_to_chroma_db(FOLDER_URL, HERE / "chroma_db")


if not os.path.isdir(DB_DIR):
    create_chroma_db()

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)


# 툴 정의
@tool(parse_docstring=True)
def role_search(question: str, role: str) -> str:
    """사내 벡터DB에서 특정 role 관련 문서를 검색합니다.

    Args:
        question: 사용자가 입력한 질문
        role: 검색할 팀/직급 (예: 'frontend', 'backend', 'data_ai', 'cto')
    """
    logger.info(f"{role} search question: {question}")
    try:
        # role 기반 검색 수행
        results = vectorstore.similarity_search(question, k=10, filter={"role": role})
        if not results:
            return f"{role} 관련된 답변을 찾지 못했습니다."

        output = []
        for i, result in enumerate(results, start=1):
            output.append(
                f"[{role}] Result {i}:\n"
                f"Content: {result.page_content}\n"
                f"Metadata: {result.metadata}"
            )
        logger.info(f"{role} search found {len(output)} results.")
        return "\n\n".join(output)
    except Exception as e:
        return f"{role} 검색 중 오류 발생: {e}"


@tool(parse_docstring=True)
def cto_search(question: str) -> str:
    """CTO 관련 질문일 경우, 모든 role(cto, backend, frontend, data_ai)을 함께 검색합니다.

    Args:
        question: 사용자가 입력한 질문
    """
    logger.info(f"CTO search question: {question}")
    outputs = []
    results = vectorstore.similarity_search(question, k=10)
    if results:
        for i, result in enumerate(results, start=1):
            outputs.append(
                f"Result {i}:\n"
                f"Content: {result.page_content}\n"
                f"Metadata: {result.metadata}"
            )
    logger.info(f"CTO search found {len(outputs)} results.")
    return "\n\n".join(outputs) if outputs else "CTO 관련된 답변을 찾지 못했습니다."


# 서비스 클래스
class LangChainChatService:
    def __init__(self):
        self.api_url = os.getenv("OLLAMA_API_URL")
        if not self.api_url:
            raise ValueError("OLLAMA_API_URL environment variable is required")

        self.model_name = os.getenv("OLLAMA_MODEL")

        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.api_url,
            api_key="ollama",
            temperature=0.2,
        )

        # 툴 바인딩 (cto, frontend, backend, data_ai 검색 지원)
        self.llm_tools = self.llm.bind_tools([role_search, cto_search])

    async def get_chat_response(self, history: list[dict]) -> str:
        try:
            system_message = """
            당신은 사내 지식을 활용하여 사용자의 질문에 정확하고 유용한 답변을 제공하는 한국인 AI 비서입니다.
            다음 지침을 따르세요:
            1. 항상 정중하고 전문적인 어조를 유지하세요.
            2. 일상적인 질문에는 일상적인 답변을 제공하세요.
            3. 사용자의 질문을 주의 깊게 읽고 이해한 후 답변
            4. 답변이 불확실한 경우, "잘 모르겠습니다"라고 솔직하게 말하세요.
            5. 정보를 활용할 때는 신뢰할 수 있는 출처를 사용하세요.
            6. 답변이 너무 길어지지 않도록 주의하세요.
            7. 필요시 툴을 사용하여 정보를 검색하세요:

            - 사용자가 CTO라면 반드시 "cto_search" 툴을 호출하세요.
            - 사용자가 frontend팀 이라면 role="frontend" 으로 "role_search" 툴을 호출하세요.
            - 사용자가 backend팀 이라면 role="backend" 으로 "role_search" 툴을 호출하세요.
            - 사용자가 data_ai팀 이라면 role="data_ai" 으로 "role_search" 툴을 호출하세요.
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
            logger.info(f"LLM Raw Response Success")
            state.append(ai)

            # 툴 호출 확인
            tool_calls = getattr(ai, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    if call["name"] == "role_search":
                        result = role_search.invoke(call["args"])
                        logger.info(f"role_search tool Response Success")
                        state.append(
                            ToolMessage(tool_call_id=call["id"], content=result)
                        )
                    elif call["name"] == "cto_search":
                        result = cto_search.invoke(call["args"])
                        logger.info(f"cto_search tool Response Success")
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


chat_service = LangChainChatService()
