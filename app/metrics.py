"""Метрики Prometheus: HTTP-метрики приложения и бизнес-метрики сервиса.

/metrics отдаёт только prometheus (политика прокси, infra/mtls/*.conf); наружу через шлюз его
нет. HTTP-метрики группируются по шаблону маршрута, а не по реальному пути, поэтому число
значений метки handler ограничено числом маршрутов.
"""

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


def setup_metrics(app: FastAPI) -> None:
    """Подключает сбор HTTP-метрик и маршрут GET /metrics."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=False)


EMBED_REQUESTS = Counter(
    "pochemuchnik_embed_requests_total",
    "Принятые запросы /embed (не отклонённые по размеру)",
)
EMBED_BATCH_SIZE = Histogram(
    "pochemuchnik_embed_batch_size",
    "Число текстов в запросе /embed",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
EMBED_DURATION = Histogram(
    "pochemuchnik_embed_duration_seconds",
    "Время кодирования текстов моделью",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
