from .langgraph_setting_edit import graph_setting_edit

graph = graph_setting_edit()


def run_langraph(user_input, config_id, chat_history=None):
    config = {"configurable": {"thread_id": config_id}}

    # chat_history가 None이면 빈 리스트로 초기화
    if chat_history is None:
        chat_history = []

    result = graph.invoke(
        {"messages": chat_history, "question": user_input}, config=config
    )

    return result
