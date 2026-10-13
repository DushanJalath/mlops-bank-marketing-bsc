"""Prediction service. Every prediction and every piece of feedback is printed as one JSON
line; Azure Container Apps ships these lines to Log Analytics, where monitoring reads them."""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from catboost import CatBoostClassifier

# Send request metrics to Application Insights when running in Azure.
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor()

from fastapi import FastAPI  # noqa: E402

from app.schemas import FeedbackRequest, PredictionRequest, PredictionResponse  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.cbm")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
STATE: dict = {}

# A dedicated logger writes one complete JSON line per event. The handler's lock stops
# lines from different request threads getting mixed together.
events = logging.getLogger("events")
events.setLevel(logging.INFO)
events.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
events.addHandler(_handler)


def log_event(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    events.info(json.dumps(event))


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    STATE["model"] = model
    STATE["features"] = list(model.feature_names_)
    yield
    STATE.clear()


app = FastAPI(title="Bank marketing propensity API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    start = time.perf_counter()
    features = req.model_dump(by_alias=True)
    X = pd.DataFrame([features])[STATE["features"]]
    probability = float(STATE["model"].predict_proba(X)[0, 1])
    response = PredictionResponse(
        request_id=str(uuid.uuid4()),
        probability=round(probability, 5),
        prediction=int(probability >= THRESHOLD),
        model_version=MODEL_VERSION,
    )
    log_event(
        {
            "event": "prediction",
            **response.model_dump(),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "features": features,
        }
    )
    return response


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    log_event({"event": "feedback", **req.model_dump()})
    return {"status": "recorded"}
