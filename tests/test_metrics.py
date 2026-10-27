from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from metrics import setup_metrics  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Метрики запросов в обработке регистрируются в общем реестре: приложение с ними одно на модуль.
    app = FastAPI()

    @app.get("/items/{item_id}")
    def get_item(item_id: int):
        return {"id": item_id}

    setup_metrics(app)
    return TestClient(app)


def test_metrics_endpoint_exposes_http_metrics_by_route_template(client):
    client.get("/items/1")
    client.get("/items/2")

    body = client.get("/metrics").text

    assert "http_request_duration_seconds" in body
    assert "http_requests_inprogress" in body
    assert (
        'http_requests_total{handler="/items/{item_id}",method="GET",status="2xx"} 2.0' in body
    )
    assert 'handler="/items/1"' not in body


def test_unknown_paths_do_not_create_new_label_values(client):
    client.get("/no/such/path/1")
    client.get("/no/such/path/2")

    assert "/no/such/path" not in client.get("/metrics").text


def test_metrics_endpoint_does_not_count_itself(client):
    client.get("/metrics")

    assert 'handler="/metrics"' not in client.get("/metrics").text
