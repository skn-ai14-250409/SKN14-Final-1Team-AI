from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from .rag import basic_chain_setting, query_setting
from .retriever import retriever_setting

basic_chain = basic_chain_setting()
retriever = retriever_setting()
query_chain = query_setting()


class ChatState(TypedDict, total=False):
    question: str  # 유저 질문
    answer: str  # 모델 답변
    rewritten: str  # 통합된 질문
    queries: List[str]  # 쿼리(질문들)
    search_results: List[str]  # 벡터 DB 검색 결과들
    messages: List[Dict[str, str]]  # 사용자 및 모델의 대화 히스토리


# (1) 사용자 질문 + 히스토리 통합 → 통합된 질문과 쿼리 추출
def extract_queries(state: ChatState) -> ChatState:
    user_text = state["question"]

    # 히스토리에서 최근 몇 개의 메시지를 가져와서 통합 질문을 생성
    messages = state.get("messages", [])

    # 최근 4개 메시지만 사용
    history_tail = messages[-4:] if messages else []
    context = history_tail.copy()

    # 현재 사용자 질문 추가
    context.append({"role": "user", "content": user_text})
    state["rewritten"] = context

    return state


# (2) LLM에게 질문 분리를 시킨다
def split_queries(state: ChatState) -> ChatState:
    rewritten = state.get("rewritten")

    response = query_chain.invoke({"rewritten": rewritten})
    state["queries"] = response["questions"]  # questions 리스트만 저장

    return state


# (3) 벡터 DB 툴 호출
def search_tool(query: str):
    """질문을 바탕으로 벡터 DB에서 결과 검색"""
    return retriever.invoke(query)  # retriever는 DB 검색 로직을 호출


# (4) 기본 답변 생성 노드
def basic_langgraph_node(state: ChatState) -> Dict[str, Any]:
    """질문에 대한 기본 답변 생성"""
    queries = state["queries"]
    history = state['messages']
    print(f"생성된 질문 리스트 {queries}")
    search_results = []

    # 각 쿼리마다 벡터 DB 검색
    for query in queries:
        print(f"{query} 검색중...")
        results = search_tool(query)
        search_results.append(results)  # 검색된 결과들을 모아서 저장

    # 검색된 결과를 바탕으로 답변 생성
    answer = basic_chain.invoke(
        {
            "question": state["question"],
            "context": "\n".join([str(res) for res in search_results]),
            "history": history,
        }
    ).strip()

    state['answer'] = answer
    state['search_results'] = search_results

    return state  # 답변을 반환
