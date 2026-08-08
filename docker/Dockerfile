FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir ".[embeddings]"
COPY scripts ./scripts
COPY data/offline_catalog.json ./data/offline_catalog.json
ENV TOWER_BACKEND=sentence_transformers \
    TOWER_MODEL_NAME=intfloat/multilingual-e5-small \
    TOWER_DEVICE=cpu
RUN python scripts/generate_offline_category_kb.py
RUN python scripts/build_demo_index.py
ENV TOWER_LOCAL_FILES_ONLY=true

FROM base AS dev
ENV AGENT_LOOP_MAX_ITERATIONS=100 \
    AGENT_TIMEOUT_SEC=600 \
    FORK_MAX_DEPTH=5 \
    LOG_LEVEL=DEBUG \
    COMPRESS_KEEP_RECENT=5
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS prod
ENV AGENT_LOOP_MAX_ITERATIONS=30 \
    AGENT_TIMEOUT_SEC=300 \
    FORK_MAX_DEPTH=2 \
    LOG_LEVEL=WARNING \
    COMPRESS_KEEP_RECENT=3
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
