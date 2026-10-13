import os

import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from src.data import CATEGORIES, FEATURES
from tests.conftest import make_rows


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny model on synthetic data so the API can be tested without the real model.
    df = make_rows(300)
    model = CatBoostClassifier(iterations=20, verbose=0, cat_features=list(CATEGORIES))
    model.fit(df[FEATURES], df["y"])
    path = tmp_path_factory.mktemp("model") / "model.cbm"
    model.save_model(str(path))
    os.environ["MODEL_PATH"] = str(path)

    import importlib

    import app.main

    importlib.reload(app.main)  # pick up MODEL_PATH
    with TestClient(app.main.app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return make_rows(1).drop(columns=["y"]).iloc[0].to_dict()


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_predict(client, payload):
    body = client.post("/predict", json=payload).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["request_id"]


def test_contract_rejects_negative_pdays(client, payload):
    payload["pdays"] = -1
    assert client.post("/predict", json=payload).status_code == 422


def test_contract_rejects_unknown_job(client, payload):
    payload["job"] = "astronaut"
    assert client.post("/predict", json=payload).status_code == 422


def test_feedback(client, payload):
    request_id = client.post("/predict", json=payload).json()["request_id"]
    response = client.post("/feedback", json={"request_id": request_id, "actual_outcome": 1})
    assert response.status_code == 200
