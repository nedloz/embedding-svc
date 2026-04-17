import json
import logging
import os

import torch
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-svc")

MODEL_NAME = os.getenv("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
SERVICE_NAME = os.getenv("SERVICE_NAME", "embedding-svc")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")
INTERNAL_AUTH_ENABLED = os.getenv("INTERNAL_AUTH_ENABLED", "true").lower() == "true"
TRUSTED_SERVICE_TOKENS_RAW = os.getenv("TRUSTED_SERVICE_TOKENS", "{}")
INTERNAL_AUTH_HEADER_NAME = os.getenv("INTERNAL_AUTH_HEADER_NAME", "X-Service-Token")
INTERNAL_SERVICE_NAME_HEADER = os.getenv("INTERNAL_SERVICE_NAME_HEADER", "X-Service-Name")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Loading model: %s", MODEL_NAME)
logger.info("Using device: %s", DEVICE)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
logger.info("Model loaded")


def trusted_tokens() -> dict[str, str]:
    try:
        return json.loads(TRUSTED_SERVICE_TOKENS_RAW)
    except Exception:
        return {}


def verify_internal_request(
    x_service_name: str = Header(..., alias=INTERNAL_SERVICE_NAME_HEADER),
    x_service_token: str = Header(..., alias=INTERNAL_AUTH_HEADER_NAME),
):
    if not INTERNAL_AUTH_ENABLED:
        return
    expected = trusted_tokens().get(x_service_name)
    if not expected or expected != x_service_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal token",
        )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


app = FastAPI()


@app.get("/ping")
def ping():
    dim = model.get_sentence_embedding_dimension()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": dim,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest,
    x_service_name: str = Header(..., alias=INTERNAL_SERVICE_NAME_HEADER),
    x_service_token: str = Header(..., alias=INTERNAL_AUTH_HEADER_NAME),
):
    verify_internal_request(x_service_name, x_service_token)

    vectors = model.encode(
        req.texts,
        normalize_embeddings=req.normalize,
        show_progress_bar=False,
    )

    result = [v.tolist() for v in vectors]
    dim = len(result[0]) if result else model.get_sentence_embedding_dimension()

    return EmbedResponse(
        model=MODEL_NAME,
        dimension=dim,
        embeddings=result,
    )