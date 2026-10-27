import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

# Модель и CUDA для проверки метрик не нужны: подменяем тяжёлые зависимости заглушками.
sys.modules.setdefault("torch", SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)))
sys.modules.setdefault("sentence_transformers", SimpleNamespace(SentenceTransformer=object))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ["INTERNAL_AUTH_MODE"] = "insecure"

import main  # noqa: E402


def sample(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def encode(self, texts, **_kwargs):
        return [FakeVector([0.0, 1.0]) for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 2


@pytest.fixture(autouse=True)
def loaded_model(monkeypatch):
    monkeypatch.setattr(main, "_model", FakeModel(), raising=False)
    monkeypatch.setattr(main, "_require_model", lambda: FakeModel())


def call_embed(texts):
    request = SimpleNamespace(headers={}, client=None)
    return main.embed(main.EmbedRequest(texts=texts), request)


def test_embed_counts_request_batch_size_and_duration():
    requests = sample("pochemuchnik_embed_requests_total")
    batches = sample("pochemuchnik_embed_batch_size_count")
    batch_sum = sample("pochemuchnik_embed_batch_size_sum")
    durations = sample("pochemuchnik_embed_duration_seconds_count")

    call_embed(["a", "b", "c"])

    assert sample("pochemuchnik_embed_requests_total") == requests + 1
    assert sample("pochemuchnik_embed_batch_size_count") == batches + 1
    assert sample("pochemuchnik_embed_batch_size_sum") == batch_sum + 3
    assert sample("pochemuchnik_embed_duration_seconds_count") == durations + 1


def test_oversized_request_is_not_counted_as_accepted():
    before = sample("pochemuchnik_embed_requests_total")
    with pytest.raises(HTTPException) as exc:
        call_embed(["x"] * (main.MAX_TEXTS_PER_REQUEST + 1))
    assert exc.value.status_code == 413
    assert sample("pochemuchnik_embed_requests_total") == before
