from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_NAME = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)
app = FastAPI(title="Globex Reranker")
_tokenizer = None
_model = None


class RerankRequest(BaseModel):
    query: str
    candidates: list[str]


class RerankResponse(BaseModel):
    scores: list[float]


def _load_model():
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        _model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_NAME)
            .eval()
        )
        if torch.cuda.is_available():
            _model = _model.half().cuda()
    return _tokenizer, _model


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
) -> RerankResponse:
    import torch

    tokenizer, model = _load_model()
    pairs = [
        (request.query, candidate)
        for candidate in request.candidates
    ]
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }
    with torch.no_grad():
        logits = model(**encoded).logits
    scores = logits.reshape(-1).float().cpu()
    return RerankResponse(
        scores=scores.tolist()
    )
