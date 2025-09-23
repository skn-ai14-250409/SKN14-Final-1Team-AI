import os
import pandas as pd
import random
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# OpenAI API 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_DIR = "./chroma_google_api_db"
OUT_FILE   = "./user_queries.csv"
NUM_QUERIES = 200
QUERIES_PER_DOC = 5
EXTRA_FACTOR = 3

# Chroma 로드
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
col = chroma_client.list_collections()[0].name
collection = chroma_client.get_collection(col)

# 문서 가져오기
docs = collection.get(include=["documents", "metadatas"])
all_ids   = docs["ids"]
all_docs  = docs["documents"]

print(f"총 문서 수: {len(all_ids)}")

# 샘플링
num_docs_needed = (NUM_QUERIES // QUERIES_PER_DOC) * EXTRA_FACTOR
sample_idxs = random.sample(range(len(all_ids)), min(num_docs_needed, len(all_ids)))

rows = []
qid_counter = 1

for idx in sample_idxs:
    doc_id = all_ids[idx]
    content = (all_docs[idx] or "")
    
    prompt = f"""
    아래는 Google API 문서의 일부입니다:

    ---
    {content}
    ---

    위 문서를 참고해서, 실제 사용자가 검색할 법한 질문을 {QUERIES_PER_DOC}개 생성해줘.
    조건:
    - 질문은 한국어/영어/혼합 표기를 자유롭게 사용
    - 짧고 간단하고 단순하게, 실제로 검색창에 칠 것처럼 작성
    - 문서에 있는 원문 단어를 그대로 쓰지 말고, 일상적인 말투나 자연스러운 한국어 표현으로 변형
    - 한국어와 영어 표기를 섞거나, 한글로 음차 표기를 사용하기도 함
    (예: 'firebase' → '파이어베이스', 'cloud storage' → '클라우드 저장소')
    - 가끔은 불완전한 문장, 띄어쓰기 오류, 맞춤법 오류도 포함
    - 한국어 발음/축약/대충 말하는 식도 섞기
    - 기술 용어를 축약하거나 대충 부르는 것도 허용
    (예: 'auth' → '로그인', 'bigquery' → '빅쿼리')
    - 실제 검색창에 타이핑하듯이 만들어라
    - 문서에 실제로 언급된 주제와 연관된 질문만 만들 것
    - 문서와 전혀 관련 없는 일상 대화는 제외
    - JSON 리스트 형식으로 출력: ["질문1", "질문2"]
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    try:
        queries = eval(resp.choices[0].message.content.strip())
    except:
        continue

    for q in queries:
        rows.append({
            "query_id": f"q{qid_counter}",
            "query_text": q
        })
        qid_counter += 1

    if len(rows) >= NUM_QUERIES:
        break

# CSV 저장
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {OUT_FILE} (총 {len(df_out)}개 쿼리)")