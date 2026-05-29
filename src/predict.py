"""
Scoring utilities:
  - predict_risk_proba   : P(default) for a customer feature vector
  - probability_to_score : maps P(default) → credit score (300–850)
  - predict_loan         : returns recommended loan amount and duration
"""

import numpy as np
import pandas as pd
import joblib

from src.data_processing import get_feature_columns


# ---------------------------------------------------------------------------
# Credit score scaling (log-odds linear transform, FICO-style)
# ---------------------------------------------------------------------------

SCORE_MIN = 300
SCORE_MAX = 850
PDO = 20          # points-to-double-odds
BASE_SCORE = 600
BASE_ODDS = 50    # good:bad odds at base score


def probability_to_score(probability: float | np.ndarray) -> float | np.ndarray:
    """Converts P(default) to a credit score in [300, 850]."""
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    odds = (1 - probability) / probability
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    score = offset + factor * np.log(odds)
    return np.clip(score, SCORE_MIN, SCORE_MAX).round().astype(int)


# ---------------------------------------------------------------------------
# Model loading (lazy, cached at module level)
# ---------------------------------------------------------------------------

_models: dict = {}


def _load(name: str, path: str):
    if name not in _models:
        _models[name] = joblib.load(path)
    return _models[name]


def get_risk_model(model_path: str = "data/processed/risk_model.pkl"):
    return _load("risk", model_path)


def get_amount_model(model_path: str = "data/processed/loan_amount_model.pkl"):
    return _load("amount", model_path)


def get_duration_model(model_path: str = "data/processed/loan_duration_model.pkl"):
    return _load("duration", model_path)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def predict_risk_proba(features: dict | pd.DataFrame, model_path: str = "data/processed/risk_model.pkl") -> float:
    model = get_risk_model(model_path)
    cols = get_feature_columns()
    if isinstance(features, dict):
        X = pd.DataFrame([features])[cols].fillna(0)
    else:
        X = features[cols].fillna(0)
    prob = model.predict_proba(X)[:, 1]
    return float(prob[0]) if len(prob) == 1 else prob


def predict_loan(
    features: dict | pd.DataFrame,
    amount_model_path: str = "data/processed/loan_amount_model.pkl",
    duration_model_path: str = "data/processed/loan_duration_model.pkl",
) -> dict:
    amount_model = get_amount_model(amount_model_path)
    duration_model = get_duration_model(duration_model_path)
    cols = get_feature_columns()
    if isinstance(features, dict):
        X = pd.DataFrame([features])[cols].fillna(0)
    else:
        X = features[cols].fillna(0)

    amount = float(amount_model.predict(X)[0])
    duration = int(round(float(duration_model.predict(X)[0])))
    return {"recommended_amount": round(amount, 2), "recommended_duration_months": duration}


def score_customer(features: dict) -> dict:
    """Full scoring pipeline: risk proba → credit score → loan recommendation."""
    prob = predict_risk_proba(features)
    score = int(probability_to_score(prob))
    loan = predict_loan(features)
    return {
        "default_probability": round(prob, 4),
        "credit_score": score,
        "risk_category": _risk_category(score),
        **loan,
    }


def _risk_category(score: int) -> str:
    if score >= 750:
        return "Very Low Risk"
    if score >= 670:
        return "Low Risk"
    if score >= 580:
        return "Medium Risk"
    if score >= 500:
        return "High Risk"
    return "Very High Risk"
