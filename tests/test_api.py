import numpy as np
import pandas as pd
import pytest
import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api.main as main_module
from src.api.pydantic_models import PredictRequest

FEATURE_COLUMNS = list(PredictRequest.model_fields.keys())
EXAMPLE_REQUEST = PredictRequest.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def stub_model_path(tmp_path):
    """A tiny, deterministically-fitted LogisticRegression standing in for the real champion model."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = np.array([0, 1] * (n // 2))  # guarantees both classes are present
    model = LogisticRegression().fit(X, y)

    path = tmp_path / "risk_model.pkl"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture()
def client(stub_model_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODEL_PATH", stub_model_path)
    monkeypatch.setattr(main_module, "_model", None)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_happy_path(client):
    resp = client.post("/predict", json=EXAMPLE_REQUEST)
    assert resp.status_code == 200

    body = resp.json()
    assert set(body.keys()) == {"default_probability", "credit_score", "risk_category"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert 300 <= body["credit_score"] <= 850
    assert body["risk_category"] in {
        "Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk",
    }


def test_predict_missing_required_field_returns_422(client):
    incomplete = dict(EXAMPLE_REQUEST)
    del incomplete["total_amount"]

    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_without_loaded_model_returns_503(monkeypatch):
    monkeypatch.setattr(main_module, "MODEL_PATH", "/nonexistent/risk_model.pkl")
    monkeypatch.setattr(main_module, "_model", None)

    with TestClient(main_module.app) as test_client:
        resp = test_client.post("/predict", json=EXAMPLE_REQUEST)

    assert resp.status_code == 503
