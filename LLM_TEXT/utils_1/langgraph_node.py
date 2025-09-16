from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from .rag import basic_chain_setting
from .retriever import retriever_setting

basic_chain = basic_chain_setting()
retriever = retriever_setting()


class ChatState(TypedDict, total=False):
    question: str  # 유저 질문
    answer: str  # 모델 답변
    search_results: List[str]  # 벡터 DB 검색 결과들



# (1) 벡터 DB 툴 호출
def search_tool(query: str):
    """질문을 바탕으로 벡터 DB에서 결과 검색"""
    return retriever.invoke(query)  # retriever는 DB 검색 로직을 호출


# (2) 기본 답변 생성 노드
def basic_langgraph_node(state: ChatState) -> Dict[str, Any]:
    search_results = []
    query = state.get('question')

    results = search_tool(query)
    search_results.append(results)  # 검색된 결과들을 모아서 저장
    state['search_results'] = search_results

    # 검색된 결과를 바탕으로 답변 생성
    answer = basic_chain.invoke(
        {
            "question": state["question"],
            "context": "\n".join([str(res) for res in search_results]),
        }
    ).strip()

    state["answer"] = answer

    return state  # 답변을 반환
