from pathlib import Path

from fastapi.testclient import TestClient

from rag_eval_bench.app import create_app
from rag_eval_bench.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def _client() -> TestClient:
    store = Store(ROOT / "data")
    return TestClient(create_app(store, config_path=ROOT / "frontedge.toml"))


def test_meta_reports_the_connected_screens() -> None:
    with _client() as client:
        response = client.get("/frontedge/api/meta")
    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["evals"] is True
    assert capabilities["datasets"] is True
    assert capabilities["runs"]["query"] is True


def test_runs_endpoint_returns_the_configured_page() -> None:
    with _client() as client:
        response = client.get("/frontedge/api/runs")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert response.headers["X-Frontedge-Total"] == "3"
    assert "logStreamUrl" in body[0]


def test_run_detail_and_examples() -> None:
    with _client() as client:
        detail = client.get("/frontedge/api/runs/exp-001")
        examples = client.get("/frontedge/api/runs/exp-001/examples?limit=2")
    assert detail.status_code == 200
    assert detail.json()["id"] == "exp-001"
    assert examples.status_code == 200
    assert len(examples.json()) == 2
    assert examples.headers["X-Frontedge-Total"] == "8"


def test_eval_comparison_endpoint() -> None:
    with _client() as client:
        response = client.get(
            "/frontedge/api/evals/qa-f1/compare",
            params={"reference": "exp-001", "comparison": "exp-002"},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["examples"]) == 8
    assert body["comparisonAggregate"] > body["referenceAggregate"]


def test_datasets_endpoint() -> None:
    with _client() as client:
        datasets = client.get("/frontedge/api/datasets")
        examples = client.get("/frontedge/api/datasets/ml-glossary/examples?limit=3")
    assert datasets.status_code == 200
    assert datasets.json()[0]["examplesCount"] == 8
    assert examples.status_code == 200
    assert len(examples.json()) == 3
    assert examples.headers["X-Frontedge-Total"] == "8"


def test_interface_is_served() -> None:
    with _client() as client:
        response = client.get("/frontedge/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
