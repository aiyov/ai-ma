# ai-ma API image (FastAPI + FastEmbed + chat tools).
# Qdrant / Redis stay as separate compose services.
#
#   docker build -t ai-ma .
#   docker compose --profile api up -d --build

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE=/app/.cache/fastembed \
    IMAGE_CACHE=/app/.cache/images \
    SYNC_STATE_PATH=/app/.cache/sync_state.sqlite \
    DATA_PATH=/app/data.json

# libgomp: onnxruntime / FastEmbed; curl: HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY web ./web
COPY data.json ./data.json

RUN mkdir -p /app/.cache/fastembed /app/.cache/images

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
