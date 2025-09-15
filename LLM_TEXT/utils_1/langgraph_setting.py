from typing import TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, START, END

from .rag import basic_chain_setting
from .retriever import retriever_setting
from langgraph.checkpoint.memory import MemorySaver
from .langgraph_node import (
    ChatState,
    basic_langgraph_node,
)


basic_chain = basic_chain_setting()
retriever = retriever_setting()


# basic_langgraph_node는 langgraph_node.py에서 import


def graph_setting():
    # LangGraph 정의
    graph = StateGraph(ChatState)

    # 노드 등록
    graph.add_node("basic", basic_langgraph_node)  # 기본 답변 노드

    # 시작 노드 정의
    graph.set_entry_point("basic")

    graph.add_edge("basic", END)  # basic에서 바로 end로 보내기

    memory = MemorySaver()
    compiled_graph = graph.compile()

    return compiled_graph
