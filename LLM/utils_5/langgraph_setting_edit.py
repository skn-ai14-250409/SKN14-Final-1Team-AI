from typing import TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, START, END

from .rag import basic_chain_setting
from .retriever import retriever_setting
from langgraph.checkpoint.memory import MemorySaver
from .langgraph_node import *


basic_chain = basic_chain_setting()
retriever = retriever_setting()


# 그래프 설정
def graph_setting_edit():
    # LangGraph 정의
    graph = StateGraph(ChatState)

    # 노드 등록
    graph.add_node("extract_queries", extract_queries)  # 질문 통합 + 쿼리 추출 노드
    graph.add_node("split_queries", split_queries)  # 질문 분리 툴
    graph.add_node("basic", basic_langgraph_node)  # 기본 답변 노드
    graph.add_node("tool", tool_based_search_node)

    # 시작 노드 정의
    graph.set_entry_point("extract_queries")

    # 흐름 설정
    graph.add_edge("extract_queries", "split_queries")  # 질문 추출 후 분리
    graph.add_edge("split_queries", "tool")  # 쿼리 분리 후 벡터 db tool노드로 넘어감
    graph.add_edge("tool", "basic") # tool노드에서 벡터 db 검색 후 답변 노드로 넘어감
    graph.add_edge("basic", END)  # 기본 답변 후 종료

    # 그래프 컴파일
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph
