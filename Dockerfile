FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 코드와 메타데이터만 복사 (모델/데이터는 docker-compose 볼륨으로 마운트)
# → 2.8GB reasoning 모델을 이미지에 굽지 않아 빌드가 빠르고 가벼움.
#   (모델이 gitignore 되어 빌드 컨텍스트에서 빠지면 COPY 가 실패하던 문제도 해소)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# PyTorch CPU 버전 + 프로젝트 의존성 설치
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

ENV PORT=8080
ENV APP_ROOT=/app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn aipa_engine.main:app --host 0.0.0.0 --port $PORT"]
