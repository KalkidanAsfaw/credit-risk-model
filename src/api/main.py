"""
FastAPI service exposing the credit risk scoring pipeline.

Endpoints:
  GET  /health          — liveness check
  POST /score           — full score for one customer
  POST /risk-probability — raw P(default) only
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerFeatures, ScoreResponse, HealthResponse
from src.predict import score_customer, predict_risk_proba, probability_to_score


MODEL_DIR = os.getenv("MODEL_DIR", "data/processed")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
AMOUNT_MODEL_PATH = os.path.join(MODEL_DIR, "loan_amount_model.pkl")
DURATION_MODEL_PATH = os.path.join(MODEL_DIR, "loan_duration_model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models at startup so first request isn't slow
    try:
        from src.predict import get_risk_model, get_amount_model, get_duration_model
        get_risk_model(RISK_MODEL_PATH)
        get_amount_model(AMOUNT_MODEL_PATH)
        get_duration_model(DURATION_MODEL_PATH)
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model files not found. Train models before serving requests.")
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Bati Bank — eCommerce credit risk model service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(customer: CustomerFeatures):
    try:
        result = score_customer(
            features=customer.model_dump(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/risk-probability")
def risk_probability(customer: CustomerFeatures):
    try:
        prob = predict_risk_proba(customer.model_dump(), model_path=RISK_MODEL_PATH)
        score = int(probability_to_score(prob))
        return {"default_probability": round(float(prob), 4), "credit_score": score}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
