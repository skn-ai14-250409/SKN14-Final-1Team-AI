from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .retriever import retriever_setting
from .retriever_qa import retriever_setting2


# 벡터DB tag 통일 후 삭제 예정
TAG_ALIAS = {
    "firebase_firestore_crawled": "firestore",
    "firebase_auth_crawled": "firebase_authentication",
}


TAG_ALIAS_QA = {
    "firestore": "firestore",
    "firebase_authentication": "firebase_authentication",
}


def normalize_tag(tag: str, is_qa: bool = False) -> str:
    """
    태그명을 표준 태그명으로 변환
    - is_qa=True: QA DB용 alias 적용
    - is_qa=False: 원문 DB용 alias 적용
    """
    if is_qa:
        return TAG_ALIAS_QA.get(tag, tag)
    return TAG_ALIAS.get(tag, tag)


def bm25_retrievers_by_tag(k=5):
    """
    원문 Chroma DB 문서를 태그별로 분리하여 BM25 retrievers 생성
    """
    vs = retriever_setting()

    # Chroma에서 문서 + 메타데이터 꺼내오기
    data = vs.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]

    # 태그별로 문서 묶기
    tag_docs = defaultdict(list)
    for doc, meta in zip(docs, metas):
        # 태그 alias 적용
        raw_tag = meta["tags"]
        tag = normalize_tag(raw_tag, is_qa=False)  # 원문 DB → 표준 태그

        tag_docs[tag].append(Document(page_content=doc, metadata=meta))

    # 태그별 BM25Retriever 생성
    bm25_dict = {}
    for tag, dlist in tag_docs.items():
        r = BM25Retriever.from_documents(dlist)
        r.k = k
        bm25_dict[tag] = r

    return bm25_dict


def bm25_retrievers_by_tag_qa(k=10):
    """
    QA Chroma DB 문서를 태그별로 분리하여 BM25 retrievers 생성
    """
    vs = retriever_setting2()
    data = vs.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]

    tag_docs = defaultdict(list)
    for doc, meta in zip(docs, metas):
        raw_tag = meta["tags"]
        tag = normalize_tag(raw_tag, is_qa=True)  # QA DB → 표준 태그

        tag_docs[tag].append(Document(page_content=doc, metadata=meta))

    bm25_dict = {}
    for tag, dlist in tag_docs.items():
        r = BM25Retriever.from_documents(dlist)
        r.k = k
        bm25_dict[tag] = r
        
    return bm25_dict