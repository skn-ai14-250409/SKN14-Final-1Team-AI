FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt만 먼저 복사하여 Docker 캐시 최적화
COPY requirements.txt .

# Python 패키지 설치 (시스템 전역에 설치)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8001

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]