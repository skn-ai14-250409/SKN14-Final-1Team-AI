from dotenv import load_dotenv
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# LangChain OpenAI
from langchain_openai import ChatOpenAI


load_dotenv()


def basic_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    basic_prompt = PromptTemplate.from_template(
        """
    당신은 api 문서 관련 전문 챗봇으로서 사용자의 질문에 정확하고 친절하게 답변해야 합니다.
    아래 제공되는 문서에 없는 내용은 절대 답변에 포함하지 말고, 문서 내에서만 답변 내용을 찾아서 제공하세요.
    만약 사용자 질문이 구글 api 문서에 대한 질문이 아니라면, 아래 문서는 무시하고 일상 질문에 대해서만 답변하세요.
    일상 질문은 전문적인 내용이 아니고, 코딩과도 관련 없고, 전문지식은 전혀 쓰지 않는 단순한 일상질문입니다.
    
    예를 들어 gitflow가 뭔지 물어보면 답변할 수 없다고 해야 합니다.
    
    문서 : {context}
    
    이전 대화 내역 : {history}

    이번 사용자 질문 : {question}
    
    추가로, 이번 사용자 질문에 이미지 분석 내용이 들어있고 사용자가 '이미지에 대한 질문을 하거나' 혹은 '이건 어때?' 와 같이 물어보다면 해당 이미지 분석 결과를 참고해서 답변해주세요.
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


       **중요**: 이전 대화 맥락을 반드시 고려해서 질문을 생성하세요.
       - 현재 질문이 이전 대화와 연관되어 있다면, 이전 맥락을 포함한 통합된 질문을 만들어주세요.

       - 예: 바로 전에 "People API 연락처 조회"에 대해 이야기하고 나서, "그럼 프로필 수정은?"이라는 질문이 나오면 "People API에서 프로필 수정 방법"으로 통합해주세요.
       - 주의사항: 이전에 "People API 연락첯 조회"에 대해 이야기하고 나서, "Firebase"와 같이 다른 api에 대한 대화 내용이 나온 후 "프로필 수정은?"이라는 질문이 나오면 마지막 대화 맥락에 맞춰서, "Firebase에서 프로필 수정 방법"과 같이 통합해야 합니다.

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


def classify_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    classification_prompt = PromptTemplate.from_template(
        """
    다음 질문을 분석하여 **주요 목적**에 따라 분류하세요.

    ## 📋 분류 기준
    api: 구글 api에 대한 질문이 주 목적
    basic: 구글 api에 대한 질문이 아닌, 일상 질문이 주 목적
    none: 일상 질문도 아니면서, 구글 api 외의 전문적인 지식에 대해서 물어보는 것이 주 목적

    ## 📌 출력 규칙
    - 오직 다음 중 하나의 단어만 출력: `api`, `basic`, `none`
    - **절대 추가 설명 없이** 해당 단어만 출력
    - 예: `api`

    최근 대화 내용:
    {context}

    이번 사용자 질문:
    {question}

    ## 🧪 예시
    질문: 구글 캘린더 API에서 이벤트를 어떻게 추가하나요?  
    정답: api

    질문: Google Drive API로 파일 권한을 수정하는 방법 알려줘  
    정답: api

    질문: Gmail API에서 특정 라벨이 붙은 메일만 가져올 수 있어?  
    정답: api

    질문: 구글 맵 API 호출하는 법 알려주고, 참고로 난 지금 배고파  
    정답: api

    질문: 오늘 날씨 어때?  
    정답: basic

    질문: 지금 몇 시야?  
    정답: basic

    질문: 오늘 날씨 어때? 그리고 딥러닝 CNN 구조 설명해줘  
    정답: none

    질문: 구글 캘린더 API 문서 보여줄 수 있어? 아, 그리고 안녕!  
    정답: api

    질문: 양자컴퓨터에서 큐비트 얽힘이 뭔지 설명해줘  
    정답: none

    질문: 딥러닝에서 Transformer 구조가 뭐야?  
    정답: none

    질문: Docker 컨테이너에서 MySQL 볼륨 마운트 하는 방법 알려줘  
    정답: none

    질문: OpenAI API랑 Google API 차이가 뭐야?  
    정답: none

    질문: Google API가 아닌 AWS S3 SDK 사용법 알려줘  
    정답: none

    ---

    질문: {question}  
    정답:
    """)

    classify_chain = classification_prompt | llm | StrOutputParser()

    return classify_chain


def simple_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    simple_prompt = PromptTemplate.from_template(
        """
    너는 사용자의 **일상적인 질문**에만 답변하는 도우미야.  

    - 일상적인 질문이면 친절하게 간단히 대답해.  
    - 구글 API 질문이나 전문적인 지식 질문이면 **"대답할 수 없어요."** 라고만 답해야 하는데, 그게 보낸 이미지가 뭔지 물어보는거나 전에 물어본게 뭐였는지 물어보는거면 대답해줘도 돼.
    

    최근 대화 내용:
    {context}

    사용자 질문:
    {question}

    ---

    답변:
    """
    )

    simple_chain = simple_prompt | llm | StrOutputParser()

    return simple_chain


def impossable_chain_setting():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    imp_prompt = PromptTemplate.from_template(
        """
    너는 사용자의 **사용자의 질문**에 대한 내용을 몰라서 답변할 수 없는 챗봇이야.  

    - 최근 대화 내용 및 사용자 질문을 인용해서, **"(사용자가 질문한 내용)은 제가 모르는 내용입니다. 일상 질문 혹은 구글 api 관련 질문만 답변드릴수 있어요"** 라고만 답해.
    (사용자가 질문한 내용)은 그대로 쓰지 말고 요약하여 잘 정제하여 답변할 때 인용하세요.

    최근 대화 내용:
    {context}

    사용자 질문:
    {question}

    ---

    답변:
    """
    )

    imp_chain = imp_prompt | llm | StrOutputParser()

    return imp_chain
