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
        docs = vectorstore.similarity_search(keyword, k=6)
        if not docs:
            logger.info(f"-------- frontend 관련된 답변을 찾지 못했습니다.")
            return f"frontend 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx + 1}]]" for idx, doc in enumerate(docs)]
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
        docs = vectorstore.similarity_search(keyword, k=6)
        if not docs:
            logger.info(f"-------- backend 관련된 답변을 찾지 못했습니다.")
            return f"backend 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx + 1}]]" for idx, doc in enumerate(docs)]
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
        docs = vectorstore.similarity_search(keyword, k=6)
        if not docs:
            logger.info(f"-------- data_ai 관련된 답변을 찾지 못했습니다.")
            return f"data_ai 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx + 1}]]" for idx, doc in enumerate(docs)]
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
        docs = vectorstore.similarity_search(keyword, k=6)
        if not docs:
            logger.info(f"-------- cto 관련된 답변을 찾지 못했습니다.")
            return f"cto 관련된 답변을 찾지 못했습니다."

        ref_text = "\n".join(
            [f"{doc.page_content} [[ref{idx + 1}]]" for idx, doc in enumerate(docs)]
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

    def get_chat_response(self, request) -> str:
        history = request['history']
        permission = request['permission']
        tone = request['tone']
        logger.info(f"-------- Chat Request - Permission: {permission}, Tone: {tone}")

        try:
            # permission별 툴 선택
            if permission == "cto":
                tool_map = {"cto_search": cto_search}
                tool_prompt = """사용자는 cto로서, 모든 팀의 문서를 열람할 수 있는 개발팀 최고 관리자입니다.
                당신은 <tools></tools> 안에 있는 tool을 호출하여 문서를 검색할 수 있습니다.
                일상적인 질문(ex: 안녕, 안녕하세요, 반가워 등)의 경우, tool 호출 없이 바로 답변하세요.

                # Tools

                You may call one or more functions to assist with the user query.

                You are provided with function signatures within <tools></tools> XML tags:
                <tools>
                {"type": "function", "function": {"name": "cto_search", "description": "사내 문서 검색을 위한 도구입니다. 대화 내역을 바탕으로 사용자가 원하는 문서를 찾고, 관련된 문서를 반환합니다.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "검색할 문서 키워드 (예: \'코드노바 API 서버 설정\')"}}, "required": ["keyword"], "additionalProperties": false}}}
                </tools>

                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
                <tool_call>
                {"name": <function-name>, "arguments": <args-json-object>}'
                </tool_call>"""
            elif permission == "frontend":
                tool_map = {"frontend_search": frontend_search}
                tool_prompt = """
                                사용자는 frontend(프론트엔드)팀에 속한 팀원입니다.
                당신은 <tools></tools> 안에 있는 tool을 호출하여 문서를 검색할 수 있습니다.
                일상적인 질문(ex: 안녕, 안녕하세요, 반가워 등)의 경우, tool 호출 없이 바로 답변하세요.

                # Tools

                You may call one or more functions to assist with the user query.

                You are provided with function signatures within <tools></tools> XML tags:
                <tools>
                {"type": "function", "function": {"name": "frontend_search", "description": "사내 문서 검색을 위한 도구입니다. 대화 내역을 바탕으로 사용자가 원하는 문서를 찾고, 관련된 문서를 반환합니다.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "검색할 문서 키워드 (예: \'코드노바 API 서버 설정\')"}}, "required": ["keyword"], "additionalProperties": false}}}
                </tools>

                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
                <tool_call>
                {"name": <function-name>, "arguments": <args-json-object>}'
                </tool_call>
                                """
            elif permission == "backend":
                tool_map = {"backend_search": backend_search}
                tool_prompt = """
                                사용자는 backend(백엔드)팀에 속한 팀원입니다.
                당신은 <tools></tools> 안에 있는 tool을 호출하여 문서를 검색할 수 있습니다.
                일상적인 질문(ex: 안녕, 안녕하세요, 반가워 등)의 경우, tool 호출 없이 바로 답변하세요.

                # Tools

                You may call one or more functions to assist with the user query.

                You are provided with function signatures within <tools></tools> XML tags:
                <tools>
                {"type": "function", "function": {"name": "backend_search", "description": "사내 문서 검색을 위한 도구입니다. 대화 내역을 바탕으로 사용자가 원하는 문서를 찾고, 관련된 문서를 반환합니다.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "검색할 문서 키워드 (예: \'코드노바 API 서버 설정\')"}}, "required": ["keyword"], "additionalProperties": false}}}
                </tools>

                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
                <tool_call>
                {"name": <function-name>, "arguments": <args-json-object>}'
                </tool_call>
                                """
            elif permission == "data_ai":
                tool_map = {"data_ai_search": data_ai_search}
                tool_prompt = """
                                사용자는 Data AI(데이터 AI)팀에 속한 팀원입니다.
                당신은 <tools></tools> 안에 있는 tool을 호출하여 문서를 검색할 수 있습니다.
                일상적인 질문(ex: 안녕, 안녕하세요, 반가워 등)의 경우, tool 호출 없이 바로 답변하세요.

                # Tools

                You may call one or more functions to assist with the user query.

                You are provided with function signatures within <tools></tools> XML tags:
                <tools>
                {"type": "function", "function": {"name": "data_ai_search", "description": "사내 문서 검색을 위한 도구입니다. 대화 내역을 바탕으로 사용자가 원하는 문서를 찾고, 관련된 문서를 반환합니다.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "검색할 문서 키워드 (예: \'코드노바 API 서버 설정\')"}}, "required": ["keyword"], "additionalProperties": false}}}
                </tools>

                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
                <tool_call>
                {"name": <function-name>, "arguments": <args-json-object>}'
                </tool_call>
                                """
            else:
                tool_map = {}
                tool_prompt = f""
            logger.info(f"-------- Tools Prompt: {tool_prompt}")

            # 톤별 가이드
            if tone == "formal":
                tone_instruction = "정중하고 사무적인 어조"
                tone_instruction2 = "잘 모르겠습니다"
            elif tone == "informal":
                tone_instruction = "가볍고 친근한 반말"
                tone_instruction2 = "잘 모르겠어"

            system_message = f"""
                        당신은 사내 지식을 활용하여 사용자의 질문에 정확하고 유용한 답변을 제공하는 코드노바의 사내 문서 AI 챗봇입니다.

                        {tool_prompt}

                        그리고 다음 지침을 반드시 따르세요:
                        1. 기존의 말투는 잊고 {tone_instruction}로 답변해야 하세요.
                        2. 대화 내역의 말투도 참고하지 말고 무조건 {tone_instruction}로 답변하세요
                        3. 사실에 기반한 정보를 사용하세요.
                        4. 사용자의 질문에 대한 답변을 문서에서 찾을 수 없을 경우, {tone_instruction2}라고 솔직하게 말하세요.
                        5. 사용자가 문서에 대한 질문이 아닌, "안녕"과 같은 일상적인 질문을 한다면 해당 내용에 대해서 적절히 답변해주세요.
                        6. 답변이 너무 길지 않게 하세요.
                        7. 사용자의 말투와 상관 없이, 반드시 {tone_instruction}로 답변해야 합니다.
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
                results = []
                for call in extra_calls:
                    tool_name = call["name"]
                    tool_func = tool_map.get(tool_name)
                    if tool_func:
                        result = tool_func.invoke(call["arguments"])
                        results.append(result)
                        tool_results.append(f"<tool_response>{result}</tool_response>")
                        logger.info(f"-------- {tool_name} tool Response Success")
                    else:
                        result = None
                        results.append(result)
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

            else:
                results = ['없음']

            # 최종 답변 정리
            assistant_reply = state[-1].content
            assistant_reply = re.sub(
                r"<think>.*?</think>", "", assistant_reply, flags=re.S
            )
            assistant_reply = assistant_reply.replace("<think>", "").strip()
            assistant_reply = assistant_reply.replace("</think>", "").strip()

            logger.info("-------- Final Assistant Reply Generated")
            logger.info(f"-------- {assistant_reply}")



            return assistant_reply, results

        except Exception as e:
            raise Exception(f"Failed to get response from AI: {str(e)}")


chat_service = LangChainChatService()
