"""
FastAPI service — Bati Bank Credit Risk Scoring API.

Endpoints
---------
GET  /health   — liveness check
POST /predict  — returns default_probability, credit_score, risk_category
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.base import BaseEstimator

from src.api.pydantic_models import PredictRequest, PredictResponse, HealthResponse
from src.predict import probability_to_score, risk_category

logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "data/processed/risk_model.pkl")
_model = None


def _load_model() -> BaseEstimator:
    """Load the champion model from the local path (saved by src/train.py)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python -m src.train' first."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        _load_model()
        logger.info("Model ready.")
    except FileNotFoundError as exc:
        logger.warning("Startup warning: %s", exc)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Bati Bank BNPL credit risk model. "
        "POST /predict to score a customer and receive a default probability, "
        "credit score, and risk category."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> dict[str, str]:
    """Liveness check — returns 200 OK when the service is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a single customer.

    Accepts the 39-feature vector produced by the feature engineering pipeline
    (Tasks 3 & 4) and returns:
    - **default_probability** — P(is_high_risk) in [0, 1]
    - **credit_score** — FICO-style score in [300, 850]
    - **risk_category** — human-readable risk band
    """
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        feature_dict = request.model_dump()
        X = pd.DataFrame([feature_dict])[model.feature_names_in_]
        prob  = float(model.predict_proba(X)[0, 1])
        score = int(probability_to_score(prob))
        return PredictResponse(
            default_probability=round(prob, 4),
            credit_score=score,
            risk_category=risk_category(score),
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
