from collections import defaultdict
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from .retriever import retriever_setting
from .retriever_qa import retriever_setting2
from .retriever_bm25 import bm25_retrievers_by_tag, bm25_retrievers_by_tag_qa

_vs = retriever_setting()
_bm25_dict = bm25_retrievers_by_tag()

_vs_qa = retriever_setting2()
_bm25_dict_qa = bm25_retrievers_by_tag_qa()


def hybrid_retriever_setting(api_tags):
    """
    특정 태그 리스트에 맞는 원문 하이브리드 retriever 생성
    - api_tags: ["drive"], ["gmail"], ["drive","calendar"] 등
    """
    filters = {}
    if api_tags:
        filters["tags"] = {"$in": api_tags}

    # Chroma retriever (필터 적용)
    chroma_retriever = _vs.as_retriever(search_kwargs={"k": 5}, filter=filters)

    # 태그별 BM25 retrievers
    # bm25_retrievers = 요청된 태그들(api_tags)에 해당하는 BM25Retriever 객체들의 리스트
    bm25_retrievers = [_bm25_dict[tag] for tag in api_tags if tag in _bm25_dict]

    if not bm25_retrievers:
        return chroma_retriever  # BM25 retriever가 없으면 Chroma만 반환

    
    if len(bm25_retrievers) == 1: # 태그가 하나라면 단일 BM25
        bm25 = bm25_retrievers[0]
    else:
        # 여러 태그 BM25 합치기 -> 앙상블
        bm25 = EnsembleRetriever(
            retrievers=bm25_retrievers,
            weights=[1 / len(bm25_retrievers)] * len(bm25_retrievers) # 동일한 가중치
        )

    # 최종 하이브리드 (Chroma + BM25)
    return EnsembleRetriever(
        retrievers=[chroma_retriever, bm25],
        weights=[0.8, 0.2]
    )


def hybrid_retriever_setting_qa(api_tags):
    """
    특정 태그 리스트에 맞는 QA 하이브리드 retriever 생성
    """
    filters = {}
    if api_tags:
        filters["tags"] = {"$in": api_tags}

    chroma_retriever = _vs_qa.as_retriever(search_kwargs={"k": 5}, filter=filters)

    bm25_retrievers = [_bm25_dict_qa[tag] for tag in api_tags if tag in _bm25_dict_qa]

    if not bm25_retrievers:
        return chroma_retriever

    if len(bm25_retrievers) == 1:
        bm25 = bm25_retrievers[0]
    else:
        bm25 = EnsembleRetriever(
            retrievers=bm25_retrievers,
            weights=[1 / len(bm25_retrievers)] * len(bm25_retrievers)
        )

    return EnsembleRetriever(
        retrievers=[chroma_retriever, bm25],
        weights=[0.8, 0.2]
    )
