from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel

# LangChain OpenAI
from langchain_openai import ChatOpenAI
import json

load_dotenv()


def basic_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    basic_prompt = PromptTemplate.from_template(
        """
    당신은 api 문서 관련 전문 챗봇으로서 사용자의 질문에 정확하고 친절하게 답변해야 합니다.
    아래 제공되는 문서에 없는 내용은 절대 답변에 포함하지 말고, 문서 내에서만 답변 내용을 찾아서 제공하세요.
    만약 사용자 질문이 구글 api 문서에 대한 질문이 아니라면, 아래 문서는 무시하고 일상 질문에 대해서만 답변하세요. 

    문서 : {context}

    사용자 이전 대화 히스토리 : {history}
    
    사용자 질문 : {question}
    """
    )

    basic_chain = basic_prompt | llm | StrOutputParser()

    return basic_chain


def query_setting():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    query_prompt = PromptTemplate.from_template(
        """
       유저의 채팅 히스토리와 현재 질문이 주어집니다. 


       **중요**: 이전 대화 맥락을 반드시 고려해서 api 관련 혹은 코딩(프로그래밍) 관련 질문만을 생성하세요.
       - 현재 질문이 이전 대화와 연관되어 있다면, 이전 맥락을 포함한 통합된 질문을 만들어주세요.
       - api 혹은 코딩(프로그래밍) 외의 일상 관련된 질문은 생성하지 마세요.
       
       - 예: 바로 전에 "People API 연락처 조회"에 대해 이야기하고 나서, "그럼 프로필 수정은?"이라는 질문이 나오면 "People API에서 프로필 수정 방법"으로 통합해주세요.
       - 주의사항: 이전에 "People API 연락처 조회"에 대해 이야기하고 나서, "Firebase"와 같이 다른 api에 대한 대화 내용이 나온 후 "프로필 수정은?"이라는 질문이 나오면 마지막 대화 맥락에 맞춰서, "Firebase에서 프로필 수정 방법"과 같이 통합해야 합니다.
       
       - 이전 대화에서 이미 답변이 나온 질문은 생성하지 마세요.
       - 질문은 1개가 될 수도 있고 여러개가 될 수도 있습니다.

       대화 히스토리: {rewritten}
       
       JSON 반환 형태:
       {{"questions": ["맥락을 고려한 통합 질문 1", "맥락을 고려한 통합 질문 2", ...]}}
       """
    )

    def parse_json(response):
        return json.loads(response.content)  # response.content 사용

    chain = query_prompt | llm | parse_json
    return chain
