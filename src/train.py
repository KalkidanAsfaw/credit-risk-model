"""
Training pipeline for three models:
  1. risk_model      — XGBoost binary classifier (P(default))
  2. loan_amount_model — XGBoost regressor (optimal loan amount)
  3. loan_duration_model — XGBoost regressor (optimal loan duration in months)

Run:
    python -m src.train --data data/raw/data.csv --output data/processed/
"""

import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    mean_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

from src.data_processing import (
    load_data, clean_data, build_features,
    prepare_modelling_data, get_feature_columns,
)


# ---------------------------------------------------------------------------
# Risk probability model
# ---------------------------------------------------------------------------

def train_risk_model(X_train, y_train, X_test, y_test):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_res == 0).sum() / (y_res == 1).sum(),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"[Risk Model] Test AUC: {auc:.4f}")
    print(classification_report(y_test, (probs >= 0.5).astype(int)))
    return model


# ---------------------------------------------------------------------------
# Loan amount / duration models
# ---------------------------------------------------------------------------

def _generate_loan_targets(features: pd.DataFrame) -> pd.DataFrame:
    """
    Derives loan amount and duration heuristically from transaction history.
    Replace with actuals when historical loan data is available.
    """
    df = features.copy()
    # Amount: 3-month average monthly spend, capped
    df["loan_amount"] = (df["total_amount"] / df["days_active"].clip(lower=1) * 90).clip(upper=500_000)
    # Duration: lower risk → longer term (6–60 months)
    risk_score = df["Recency"].rank(pct=True) - df["Frequency"].rank(pct=True)
    df["loan_duration"] = (6 + (1 - risk_score.clip(0, 1)) * 54).round().astype(int)
    return df


def train_loan_models(X_train, y_train_amount, y_train_duration, X_test, y_test_amount, y_test_duration):
    amount_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    amount_model.fit(X_train, y_train_amount)
    amount_preds = amount_model.predict(X_test)
    print(f"[Loan Amount] MAE: {mean_absolute_error(y_test_amount, amount_preds):.2f}  R²: {r2_score(y_test_amount, amount_preds):.4f}")

    duration_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    duration_model.fit(X_train, y_train_duration)
    duration_preds = duration_model.predict(X_test)
    print(f"[Loan Duration] MAE: {mean_absolute_error(y_test_duration, duration_preds):.2f}  R²: {r2_score(y_test_duration, duration_preds):.4f}")

    return amount_model, duration_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and cleaning data...")
    raw = load_data(data_path)
    df = clean_data(raw)

    print("Building features and proxy label...")
    features = build_features(df)
    X, y = prepare_modelling_data(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training risk model...")
    risk_model = train_risk_model(X_train, y_train, X_test, y_test)
    joblib.dump(risk_model, os.path.join(output_dir, "risk_model.pkl"))

    print("Training loan models...")
    features_with_targets = _generate_loan_targets(features)
    feat_cols = get_feature_columns()
    X_all = features_with_targets[feat_cols].fillna(0)
    y_amount = features_with_targets["loan_amount"]
    y_duration = features_with_targets["loan_duration"]

    X_tr, X_te, ya_tr, ya_te, yd_tr, yd_te = train_test_split(
        X_all, y_amount, y_duration, test_size=0.2, random_state=42
    )
    amount_model, duration_model = train_loan_models(X_tr, ya_tr, yd_tr, X_te, ya_te, yd_te)
    joblib.dump(amount_model, os.path.join(output_dir, "loan_amount_model.pkl"))
    joblib.dump(duration_model, os.path.join(output_dir, "loan_duration_model.pkl"))

    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed/", help="Output directory for model files")
    args = parser.parse_args()
    main(args.data, args.output)
