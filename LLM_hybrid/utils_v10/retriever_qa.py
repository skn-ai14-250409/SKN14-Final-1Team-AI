import os
from dotenv import load_dotenv

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .vector_db_qa import create_chroma_db

# .env 로드
load_dotenv()

# 경로/모델 설정
HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "qa_chroma_db")
COLLECTION_NAME = "qna_collection"
EMBED_MODEL = "BAAI/bge-m3"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},  # DB 생성 시 설정과 일치해야 함
)


def retriever_setting2(force_download=False):
    # 디렉토리 존재 및 내용 확인
    need_download = force_download

    if not os.path.isdir(DB_DIR):
        need_download = True
    else:
        # 디렉토리는 있지만 내용이 불완전한지 확인
        chroma_file = os.path.join(DB_DIR, "chroma.sqlite3")
        folder_dir = os.path.join(DB_DIR, "66c170c0-0369-4132-a6c5-19f6643bf942")

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

    return vs
