import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

# Модель и CUDA не нужны: подменяем тяжёлые зависимости заглушками.
sys.modules.setdefault("torch", SimpleNamespace(
    cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)))
sys.modules.setdefault("sentence_transformers", SimpleNamespace(SentenceTransformer=object))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ["INTERNAL_AUTH_MODE"] = "insecure"

import main  # noqa: E402

E5 = "intfloat/multilingual-e5-base"


def test_query_prefix_for_e5():
    assert main.apply_prefix(["а"], "query", E5) == ["query: а"]


def test_passage_prefix_for_e5():
    assert main.apply_prefix(["а"], "passage", E5) == ["passage: а"]


def test_e5_detected_case_insensitively():
    assert main.apply_prefix(["а"], "query", "Intfloat/Multilingual-E5-Base") == ["query: а"]


def test_no_prefix_for_other_models():
    assert main.apply_prefix(["а"], "query", "BAAI/bge-m3") == ["а"]


def test_kind_defaults_to_passage():
    assert main.EmbedRequest(texts=["а"]).kind == "passage"


def test_invalid_kind_rejected():
    with pytest.raises(ValidationError):
        main.EmbedRequest(texts=["а"], kind="doc")


def test_invalid_kind_gives_422_over_http():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Отдельное приложение с той же моделью запроса: метрики main.app в общем реестре
    # конфликтуют с test_metrics.py при повторной инициализации.
    probe = FastAPI()

    @probe.post("/embed")
    def embed(req: main.EmbedRequest):
        return {"kind": req.kind}

    response = TestClient(probe).post("/embed", json={"texts": ["а"], "kind": "doc"})
    assert response.status_code == 422
    ok = TestClient(probe).post("/embed", json={"texts": ["а"], "kind": "query"})
    assert ok.json() == {"kind": "query"}


class FakeVector(list):
    def tolist(self):
        return list(self)


class RecordingModel:
    def __init__(self):
        self.seen = None

    def encode(self, texts, **_kwargs):
        self.seen = list(texts)
        return [FakeVector([0.0, 1.0]) for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 2


def call_embed(monkeypatch, req):
    model = RecordingModel()
    monkeypatch.setattr(main, "_require_model", lambda: model)
    monkeypatch.setattr(main, "MODEL_NAME", E5)
    main.embed(req, SimpleNamespace(headers={}, client=None))
    return model


def test_embed_encodes_prefixed_texts(monkeypatch):
    model = call_embed(monkeypatch, main.EmbedRequest(texts=["а"], kind="query"))
    assert model.seen == ["query: а"]


def test_length_limit_uses_unprefixed_text(monkeypatch):
    monkeypatch.setattr(main, "MAX_TEXT_CHARS", 5)
    monkeypatch.setattr(main, "MAX_TOTAL_CHARS", 5)
    # 5 символов проходят лимит, хотя с префиксом "passage: " было бы 14
    model = call_embed(monkeypatch, main.EmbedRequest(texts=["12345"]))
    assert model.seen == ["passage: 12345"]
    with pytest.raises(HTTPException) as exc:
        call_embed(monkeypatch, main.EmbedRequest(texts=["123456"]))
    assert exc.value.status_code == 413
