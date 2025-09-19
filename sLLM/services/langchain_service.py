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
)
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from models.chat_model import ChatRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 구글 드라이브 폴더 링크
DRIVE_URLS = {
    "cto": "https://drive.google.com/drive/folders/1STUOUcZWZatvaK54_B9qxv0mEEjYAub0",
    "backend": "https://drive.google.com/drive/folders/1WE9G9OxVghL26-4_CsXjbXmrv_4hi1Px?usp=sharing",
    "frontend": "https://drive.google.com/drive/folders/1SJYDdkSrHSKsy4ZoMTt-3-8tweAR6_A1?usp=sharing",
    "data_ai": "https://drive.google.com/drive/folders/1i6RymMjAWqmAkOd3o2JhikI6sQ5JxKPj?usp=sharing",
}

HERE = Path(__file__).resolve().parent
DB_DIR = HERE / "chroma_db"


def download_drive_folder_to_chroma_db(folder_url: str, target_dir: Path):
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
    DB_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in DRIVE_URLS.items():
        subdir = DB_DIR / name
        if not subdir.exists():
            logger.info(f"-------- Downloading {name} ...")
            download_drive_folder_to_chroma_db(url, subdir)
        else:
            logger.info(f"-------- {name} already exists")


if not os.path.isdir(DB_DIR):
    create_chroma_db()

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


# 툴 정의
@tool(parse_docstring=True)
def frontend_search(keyword: str) -> str:
    """사내 벡터DB에서 특정 team 관련 문서를 검색합니다.

    Args:
        keyword: 사용자가 입력한 질문
    """
    logger.info(f"-------- frontend search keyword: {keyword}")
    try:
        # frontend team 기반 검색 수행
        vectorstore = Chroma(
            persist_directory=DB_DIR / "frontend", embedding_function=embedding_model
        )
        docs = vectorstore.similarity_search(keyword, k=3)
        if not docs:
            logger.info(f"-------- frontend 관련된 답변을 찾지 못했습니다.")
            return f"frontend 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx+1}]]" for idx, doc in enumerate(docs)]
        )
        ref_text = f"검색 결과:\n-----\n{ref_text}"

        logger.info(f"-------- frontend search found {len(docs)} results.")
        return ref_text
    except Exception as e:
        return f"frontend 검색 중 오류 발생: {e}"


@tool(parse_docstring=True)
def backend_search(keyword: str) -> str:
    """사내 벡터DB에서 특정 team 관련 문서를 검색합니다.

    Args:
        keyword: 사용자가 입력한 질문
    """
    logger.info(f"-------- backend search keyword: {keyword}")
    try:
        # backend team 기반 검색 수행
        vectorstore = Chroma(
            persist_directory=DB_DIR / "backend", embedding_function=embedding_model
        )
        docs = vectorstore.similarity_search(keyword, k=3)
        if not docs:
            logger.info(f"-------- backend 관련된 답변을 찾지 못했습니다.")
            return f"backend 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx+1}]]" for idx, doc in enumerate(docs)]
        )
        ref_text = f"검색 결과:\n-----\n{ref_text}"

        logger.info(f"-------- backend search found {len(docs)} results.")
        return ref_text
    except Exception as e:
        return f"backend 검색 중 오류 발생: {e}"


@tool(parse_docstring=True)
def data_ai_search(keyword: str) -> str:
    """사내 벡터DB에서 특정 team 관련 문서를 검색합니다.

    Args:
        keyword: 사용자가 입력한 질문
    """
    logger.info(f"-------- data_ai search keyword: {keyword}")
    try:
        # data_ai team 기반 검색 수행
        vectorstore = Chroma(
            persist_directory=DB_DIR / "data_ai", embedding_function=embedding_model
        )
        docs = vectorstore.similarity_search(keyword, k=3)
        if not docs:
            logger.info(f"-------- data_ai 관련된 답변을 찾지 못했습니다.")
            return f"data_ai 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx+1}]]" for idx, doc in enumerate(docs)]
        )
        ref_text = f"검색 결과:\n-----\n{ref_text}"

        logger.info(f"-------- data_ai search found {len(docs)} results.")
        return ref_text
    except Exception as e:
        return f"data_ai 검색 중 오류 발생: {e}"


@tool(parse_docstring=True)
def cto_search(keyword: str) -> str:
    """CTO 관련 질문일 경우, cto와 모든 team(backend, frontend, data_ai)을 함께 검색합니다.

    Args:
        keyword: 사용자가 입력한 질문
    """
    logger.info(f"-------- cto search keyword: {keyword}")
    try:
        # cto 기반 검색 수행
        vectorstore = Chroma(
            persist_directory=DB_DIR / "cto", embedding_function=embedding_model
        )
        docs = vectorstore.similarity_search(keyword, k=3)
        if not docs:
            logger.info(f"-------- cto 관련된 답변을 찾지 못했습니다.")
            return f"cto 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx+1}]]" for idx, doc in enumerate(docs)]
        )
        ref_text = f"검색 결과:\n-----\n{ref_text}"

        logger.info(f"-------- cto search found {len(docs)} results.")
        return ref_text
    except Exception as e:
        return f"cto 검색 중 오류 발생: {e}"


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

    async def get_chat_response(self, request: ChatRequest) -> str:
        history = request.history
        permission = request.permission
        tone = request.tone
        logger.info(f"-------- Chat Request - Permission: {permission}, Tone: {tone}")

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
                tool_map = {"cto_search": cto_search}
                tool_prompt = f"사용자는 cto이며 반드시 cto_search 툴을 호출하세요."
            elif permission == "frontend":
                tool_map = {"frontend_search": frontend_search}
                tool_prompt = f"사용자는 {permission}팀이며 반드시 frontend_search 툴을 호출하세요."
            elif permission == "backend":
                tool_map = {"backend_search": backend_search}
                tool_prompt = f"사용자는 {permission}팀이며 반드시 backend_search 툴을 호출하세요."
            elif permission == "data_ai":
                tool_map = {"data_ai_search": data_ai_search}
                tool_prompt = f"사용자는 {permission}팀이며 반드시 data_ai_search 툴을 호출하세요."
            else:
                tool_map = {}
                tool_prompt = f""
            logger.info(f"-------- Tools Prompt: {tool_prompt}")

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

            llm_tool = self.llm.invoke(state)
            logger.info("-------- LLM Tool Parse Response Success")
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
                except json.JSONDecodeError as e:
                    logger.warning(f"Tool call JSON decode 실패: {m} ({e})")
            logger.info(f"-------- LLM Tools Match : {len(extra_calls)}")

            if extra_calls:
                tool_results = []
                for call in extra_calls:
                    tool_name = call["name"]
                    tool_func = tool_map.get(tool_name)
                    if tool_func:
                        result = tool_func.invoke(call["arguments"])
                        tool_results.append(f"<tool_response>{result}</tool_response>")
                        logger.info(f"-------- {tool_name} tool Response Success")
                    else:
                        result = None
                        tool_results.append(f"<tool_response>{result}</tool_response>")
                        logger.info(f"-------- {tool_name} tool Response Fail")

                # 여러 개 결과를 하나의 메시지로 합치기
                combined_result = "\n".join(tool_results)
                state.append(
                    HumanMessage(
                        content=combined_result,
                    )
                )
                # 툴 결과 반영 후 재호출
                llm_res = self.llm.invoke(state)
                state.append(llm_res)

            # 최종 답변 정리
            assistant_reply = state[-1].content
            assistant_reply = re.sub(
                r"<think>.*?</think>", "", assistant_reply, flags=re.S
            )
            assistant_reply = assistant_reply.replace("<think>", "").strip()
            assistant_reply = assistant_reply.replace("</think>", "").strip()

            logger.info("-------- Final Assistant Reply Generated")
            logger.info(f"-------- {assistant_reply}")

            # 제목 요약 생성
            title_llm = ChatOpenAI(
                model=self.model_name,
                openai_api_base=self.api_url,
                openai_api_key=self.api_key,
                temperature=0.0,
            )

            title_system_prompt = f"""
                다음 문장을 바탕으로 한국어로 **짧고 간결한 대화 제목**을 하나만 만들어라.
                절대 원문 문장을 그대로 복사하지 말고, 핵심 주제를 명사 중심으로 추출하라.

                규칙:                      
                - 글자 수: 12자 이상, 24자 이하
                - 반드시 명사/주제어 위주 (불필요한 수식어 제거)
                - 이모지, 따옴표, 마침표, 물음표, 느낌표, 특수문자 금지
                - 접두사·접미사, 괄호, 콜론 금지
                - 문장 그대로 복사 후 붙여넣기 하지 말고, 핵심 키워드만 뽑아서 제목화
                - 질문 원문을 절대 그대로 베끼지 말 것 (핵심 개념만 압축)
                - 답변은 오직 제목 텍스트만 출력 (불필요한 설명·접두어 금지)
                                        
                예시:
                - 입력: "코드노바의 API 서버 기술스택알려줘"
                    출력: 코드노바의 API 서버 기술스택
                - 입력: "코드노바의 캐시 만료 시간은 어떤 기준으로 설정해야 하나요?"
                    출력: 코드노바의 캐시 만료 시간
            """

            title_prompt = [
                SystemMessage(content=title_system_prompt),
                HumanMessage(
                    content=json.dumps(
                        history + [{"role": "assistant", "content": assistant_reply}],
                        ensure_ascii=False,
                    )
                ),
            ]

            title_res = title_llm.invoke(title_prompt)
            title = title_res.content.strip()
            title = re.sub(r"<think>.*?</think>", "", title, flags=re.S)
            title = title.replace("</think>", "").strip()

            return assistant_reply, title

        except Exception as e:
            raise Exception(f"Failed to get response from AI: {str(e)}")


chat_service = LangChainChatService()
