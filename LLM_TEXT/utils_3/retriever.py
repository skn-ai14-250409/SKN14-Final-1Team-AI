import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .vector_db import create_chroma_db

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# .env 로드
load_dotenv()

# 경로/모델 설정
HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "google_api_docs"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 8

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},  # DB 생성 시 설정과 일치해야 함
)


def retriever_setting(force_download=False):
    # 디렉토리 존재 및 내용 확인
    need_download = force_download

    if not os.path.isdir(DB_DIR):
        need_download = True
    else:
        # 디렉토리는 있지만 내용이 불완전한지 확인
        chroma_file = os.path.join(DB_DIR, "chroma.sqlite3")
        folder_dir = os.path.join(DB_DIR, "8013b0ca-2294-4f8f-9494-65628bc6fc3f")

        if not os.path.exists(chroma_file) or not os.path.exists(folder_dir):
            need_download = True
        else:
            # 폴더 내부 내용도 확인
            try:
                folder_contents = os.listdir(folder_dir)
                if len(folder_contents) == 0:
                    need_download = True
            except Exception:
                need_download = True

    if need_download:
        create_chroma_db()

    # 기존 크로마 벡터스토어 로드
    vs = Chroma(
        collection_name=COLLECTION_NAME,  # DB 생성 시 컬렉션명과 동일해야 함
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    metadata_field_info = [
        AttributeInfo(
            name="tags",
            type="list<string>",
            description=(
                "이 필드는 구글 API 종류를 선택하는 필드입니다. "
                "다음 11개 Google API 중에서 질문과 관련 있는 API를 1개 이상 선택해야 합니다. "
                "선택할 수 있는 API 종류는 다음과 같습니다:"
                "\n\n1. map: Google Maps API (구글 맵 API)"
                "\n2. firestore: Google Firestore API (구글 파이어스토어 API)"
                "\n3. drive: Google Drive API (구글 드라이브 API)"
                "\n4. firebase_authentication: Google Firebase API (구글 파이어베이스 API)"
                "\n5. gmail: Gmail API (구글 메일 API)"
                "\n6. google_identity: Google Identity API (구글 인증 API)"
                "\n7. calendar: Google Calendar API (구글 캘린더 API)"
                "\n8. bigquery: Google BigQuery API (구글 빅쿼리 API)"
                "\n9. sheets: Google Sheets API (구글 시트 API)"
                "\n10. people: Google People API (구글 피플 API)"
                "\n11. youtube: YouTube API (구글 유튜브 API)"
                "\n\n**주의:** 각 API는 구글의 서비스별로 제공하는 API입니다. "
                "질문에 언급된 기능이나 요구사항에 맞는 API를 정확하게 선택해주세요. "
                "예를 들어, '일정을 추가하려면?' 이라는 질문에 대해선 'calendar'를 선택해야 합니다. "
                "이 필드는 여러 개의 API를 선택할 수 있으며, 주어진 질문과 가장 관련이 있는 API를 선택해 주세요."
            ),
        ),
        AttributeInfo(
            name="chroma:document", type="string", description="문서 본문 내용"
        ),
    ]

    # SelfQueryRetriever 객체생성

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        vectorstore=vs,
        document_contents="chroma:document",  # 문서 내용을 가리키는 메타데이터 필드명
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": TOP_K},
    )

    return self_query_retriever
