FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY training/models/embedding/ ./training/models/embedding/
COPY training/models/reasoning/merged/ ./training/models/reasoning/merged/

# Install PyTorch CPU-only (경량, ~200MB) + 프로젝트 의존성
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Cloud Run uses PORT env (default 8080)
ENV PORT=8080
ENV APP_ROOT=/app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn aipa_engine.main:app --host 0.0.0.0 --port $PORT"]
