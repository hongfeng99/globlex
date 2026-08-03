FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .
COPY scripts ./scripts
COPY data/demo_items.json ./data/demo_items.json
RUN python scripts/build_demo_index.py

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
