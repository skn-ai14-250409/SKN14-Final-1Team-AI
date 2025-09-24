from .langgraph_setting2 import graph_setting

graph = graph_setting()


def run_langraph(user_input, config_id, image, chat_history=None):
    try:
        config = {"configurable": {"thread_id": config_id}}

        # chat_history가 None이면 빈 리스트로 초기화
        if chat_history is None:
            chat_history = []

        print(f"run_langraph 호출 - 입력: {user_input}, 이미지: {bool(image)}")

        result = graph.invoke(
            {"messages": chat_history, 
             "question": user_input, 
             "image": image,
             "hyde_retry": 0,  
             "pseudo_doc": "",
             "answer_quality": "",
            },
            config=config,
        )

        print(f"그래프 실행 결과: {result}")
        return result
    except Exception as e:
        print(f"run_langraph 에러: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"처리 중 오류가 발생했습니다: {str(e)}"
