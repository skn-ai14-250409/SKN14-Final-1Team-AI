# Contributing Guidelines

이 프로젝트의 실행을 위한 가이드라인 입니다.

# 실행방법

가상환경 설정하기

```python
conda create -n co-sllm python=3.11 -y
conda activate co-sllm
pip install -r requirements.txt

# black 코드 포멧팅 설정
pre-commit install
```

`.env` 설정하기

```
OLLAMA_API_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:0.6b
```

# 로컬 실행 방법

```bash
uvicorn main:app --reload --port 8001
```

# API 호출방법

```sh
# get
curl -X GET "http://127.0.0.1:8001/"

# post
curl -X POST "http://localhost:8001/api/v1/chat" \
-H "Content-Type: application/json" \
-d '{"history": [
    {"type": "user", "content": "안녕?"}]}'
```