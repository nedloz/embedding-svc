import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

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

# Заглушки из .env.example и слишком короткие значения не годятся в качестве токена:
# это единственное, что защищает внутренние API.
WEAK_TOKEN_MARKERS = ("super-secret-token", "change-me", "change_me", "changeme")
MIN_TOKEN_LENGTH = 24

# Лимиты запроса. Сервис — единственный источник эмбеддингов на всю систему: его падение
# останавливает и индексацию, и поиск, поэтому один большой запрос не должен иметь
# возможности исчерпать память. Потребители обязаны резать данные на батчи
# (см. EMBED_BATCH_SIZE в ingest-worker).
MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "256"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "1000000"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Пауза между попытками загрузки модели: начинается с MODEL_LOAD_RETRY_WAIT_SEC и
# удваивается до MODEL_LOAD_RETRY_MAX_WAIT_SEC.
MODEL_LOAD_RETRY_WAIT_SEC = float(os.getenv("MODEL_LOAD_RETRY_WAIT_SEC", "5"))
MODEL_LOAD_RETRY_MAX_WAIT_SEC = float(os.getenv("MODEL_LOAD_RETRY_MAX_WAIT_SEC", "300"))

# Модель грузится в фоновом потоке после старта процесса, а не при импорте модуля.
# Раньше неудачная загрузка (нет сети к HF, нет места, недоступен GPU) роняла импорт, и
# контейнер уходил в цикл перезапусков, а пока модель грузилась, сервис не отвечал вообще.
# Теперь процесс жив сразу: /health отвечает всегда, /ping и /embed — 503 до загрузки.
_model: SentenceTransformer | None = None
_load_attempts = 0
_load_error: str | None = None


def _load_model_forever() -> None:
    global _model, _load_attempts, _load_error
    wait = MODEL_LOAD_RETRY_WAIT_SEC
    logger.info("Using device: %s", DEVICE)
    while _model is None:
        _load_attempts += 1
        logger.info("Loading model: %s (attempt %d)", MODEL_NAME, _load_attempts)
        try:
            _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model load failed, retrying in %g s", wait)
            time.sleep(wait)
            wait = min(wait * 2, MODEL_LOAD_RETRY_MAX_WAIT_SEC)
    _load_error = None
    logger.info("Model loaded")


def _require_model() -> SentenceTransformer:
    """Загруженная модель или 503 с причиной — для readiness и рабочих эндпоинтов."""
    if _model is None:
        detail = f"Модель {MODEL_NAME} ещё загружается (попытка {_load_attempts})"
        if _load_error:
            detail += f"; последняя ошибка: {_load_error}"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return _model


def _parse_trusted_tokens() -> dict[str, str]:
    """Разбор TRUSTED_SERVICE_TOKENS один раз при запуске.

    Раньше JSON парсился на каждый запрос, а ошибка разбора молча превращалась в пустой
    словарь — то есть опечатка в переменной окружения приводила к 403 на все вызовы без
    единого указания на настоящую причину. Теперь некорректный JSON роняет сервис на старте.
    """
    try:
        parsed = json.loads(TRUSTED_SERVICE_TOKENS_RAW or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON в TRUSTED_SERVICE_TOKENS: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("TRUSTED_SERVICE_TOKENS должен быть JSON-объектом {имя: токен}")
    if INTERNAL_AUTH_ENABLED:
        weak = [
            name
            for name, token in parsed.items()
            if len(str(token)) < MIN_TOKEN_LENGTH
            or any(marker in str(token).lower() for marker in WEAK_TOKEN_MARKERS)
        ]
        if weak:
            raise RuntimeError(
                f"Слабые токены в TRUSTED_SERVICE_TOKENS для: {', '.join(weak)}. "
                f"Нужны случайные значения длиной не менее {MIN_TOKEN_LENGTH} символов "
                "(scripts/gen_service_tokens.py в корне репозитория main)."
            )
    return parsed


TRUSTED_TOKENS = _parse_trusted_tokens()

if INTERNAL_AUTH_ENABLED and not TRUSTED_TOKENS:
    logger.warning(
        "TRUSTED_SERVICE_TOKENS пуст при включённой аутентификации — /embed будет отвечать "
        "403 на любой запрос."
    )


def trusted_tokens() -> dict[str, str]:
    return TRUSTED_TOKENS


def verify_internal_request(
    x_service_name: str = Header(..., alias=INTERNAL_SERVICE_NAME_HEADER),
    x_service_token: str = Header(..., alias=INTERNAL_AUTH_HEADER_NAME),
):
    if not INTERNAL_AUTH_ENABLED:
        return
    expected = trusted_tokens().get(x_service_name)
    if not expected or not hmac.compare_digest(expected.encode(), x_service_token.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal token",
        )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    normalize: bool = True


def _check_request_size(texts: list[str]) -> None:
    """Отклоняет заведомо неподъёмные запросы до того, как они дойдут до модели."""
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Слишком много текстов в одном запросе: {len(texts)} > {MAX_TEXTS_PER_REQUEST}. "
                f"Разбейте на батчи."
            ),
        )

    total = 0
    for index, text in enumerate(texts):
        length = len(text)
        if length > MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Текст #{index} длиной {length} символов превышает лимит {MAX_TEXT_CHARS}.",
            )
        total += length

    if total > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Суммарная длина текстов {total} превышает лимит {MAX_TOTAL_CHARS}.",
        )


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Поток, а не asyncio-задача: загрузка блокирующая и не должна занимать event loop,
    # иначе /health перестанет отвечать на время загрузки.
    threading.Thread(target=_load_model_forever, name="model-loader", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness: процесс жив. Отвечает 200 и во время загрузки модели, и при её сбоях."""
    return {
        "status": "ready" if _model is not None else "loading",
        "model": MODEL_NAME,
        "attempts": _load_attempts,
        "last_error": _load_error,
    }


@app.get("/ping")
def ping():
    """Readiness: 503, пока модель не загружена. На него опирается healthcheck в compose."""
    model = _require_model()
    dim = model.get_sentence_embedding_dimension()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": dim,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }


# Размер батча при кодировании. Без явного значения sentence-transformers берёт 32 текста
# разом; у bge-m3 окно 8192 токена, поэтому пик активаций на таком батче огромен. На стенде
# это выражалось так: после индексации 676 чанков процесс держал 10 ГБ из 12 ГБ видеопамяти,
# ollama не смогла разместить модель и уехала целиком в CPU (0 из 33 слоёв на GPU).
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "8"))

# Возвращать ли освобождённую видеопамять драйверу после запроса. Кэширующий аллокатор
# PyTorch по умолчанию держит освобождённые блоки у себя — это быстрее, но на одной карте
# с ollama означает, что пик потребления навсегда остаётся занятым. Стоит несколько
# миллисекунд на запрос и заметно меньше, чем само кодирование.
RELEASE_GPU_CACHE = os.getenv("RELEASE_GPU_CACHE", "true").lower() == "true"


def _release_gpu_cache() -> None:
    if RELEASE_GPU_CACHE and DEVICE == "cuda":
        torch.cuda.empty_cache()


@app.post("/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest,
    x_service_name: str = Header(..., alias=INTERNAL_SERVICE_NAME_HEADER),
    x_service_token: str = Header(..., alias=INTERNAL_AUTH_HEADER_NAME),
):
    verify_internal_request(x_service_name, x_service_token)
    _check_request_size(req.texts)
    model = _require_model()

    vectors = model.encode(
        req.texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=req.normalize,
        show_progress_bar=False,
    )

    result = [v.tolist() for v in vectors]

    _release_gpu_cache()
    dim = len(result[0]) if result else model.get_sentence_embedding_dimension()

    return EmbedResponse(
        model=MODEL_NAME,
        dimension=dim,
        embeddings=result,
    )