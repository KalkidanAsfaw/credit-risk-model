"""
train.py  –  Model training and MLflow experiment tracking.

Trains four classifiers on the credit risk proxy target (is_high_risk):
  1. Logistic Regression  (interpretable baseline)
  2. Decision Tree
  3. Random Forest
  4. LightGBM              (gradient boosting challenger)

Each model is tracked in MLflow with parameters, metrics, and a saved artifact.
The best model (highest ROC-AUC on the test set) is registered in the
MLflow Model Registry under the name  "credit-risk-champion".

Usage
-----
  python -m src.train                            # use defaults
  python -m src.train --data data/raw/data.csv  # regenerate features first
  python -m src.train --experiment my-exp --no-tuning
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import (
    fit_full_pipeline,
    get_feature_columns,
    load_data,
    prepare_modelling_data,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_features(
    features_path: str,
    raw_path: str,
) -> pd.DataFrame:
    """
    Returns the processed feature DataFrame.
    Uses cached CSV if it exists; otherwise runs the full pipeline.
    """
    if os.path.exists(features_path):
        logger.info("Loading cached features from %s", features_path)
        return pd.read_csv(features_path)

    logger.info("features.csv not found — running full pipeline from %s", raw_path)
    raw = load_data(raw_path)
    _, features = fit_full_pipeline(raw, output_path=features_path)
    return features


def prepare_splits(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_smote: bool = True,
) -> tuple:
    """
    Splits features into train / test sets.
    Optionally applies SMOTE to the training set to address class imbalance.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = prepare_modelling_data(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split: train=%d  test=%d  positive_rate_train=%.1f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )

    if use_smote:
        minority_count = int((y_train == 1).sum())
        k_neighbors    = min(5, minority_count - 1)
        if k_neighbors >= 1:
            sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        else:
            logger.warning("SMOTE skipped: too few minority samples (%d)", minority_count)
        logger.info(
            "After SMOTE: train=%d  positive_rate=%.1f%%",
            len(X_train), y_train.mean() * 100,
        )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Returns accuracy, precision, recall, f1, roc_auc."""
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),                    4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0),  4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0),  4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0),  4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                    4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions and hyperparameter grids
# ─────────────────────────────────────────────────────────────────────────────

def get_model_configs() -> list[dict]:
    """
    Returns a list of model configurations.
    Each entry has: name, model, param_grid, log_fn.
    """
    return [
        {
            "name": "logistic_regression",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]),
            "param_grid": {
                "clf__C":       [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l1", "l2"],
                "clf__solver":  ["liblinear"],
            },
            "log_fn": mlflow.sklearn.log_model,
        },
        {
            "name": "decision_tree",
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "max_depth":        [3, 5, 10, None],
                "min_samples_leaf": [1, 5, 10],
                "criterion":        ["gini", "entropy"],
            },
            "log_fn": mlflow.sklearn.log_model,
        },
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators":     [100, 200, 300],
                "max_depth":        [5, 10, None],
                "min_samples_leaf": [1, 5],
                "max_features":     ["sqrt", "log2"],
            },
            "log_fn": mlflow.sklearn.log_model,
        },
        {
            "name": "lightgbm",
            "model": LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "param_grid": {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves":    [31, 63, 127],
                "subsample":     [0.7, 0.9, 1.0],
            },
            "log_fn": mlflow.sklearn.log_model,
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Training + MLflow tracking
# ─────────────────────────────────────────────────────────────────────────────

def train_and_track(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    experiment_name: str,
    use_tuning: bool = True,
    n_iter: int = 20,
) -> tuple[object, dict, str]:
    """
    Trains one model configuration with optional RandomizedSearchCV,
    logs everything to MLflow, and returns (best_estimator, metrics, run_id).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=config["name"]) as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_name", config["name"])
        mlflow.set_tag("smote", str(use_tuning))

        # ── Hyperparameter tuning ────────────────────────────────────────
        if use_tuning and config["param_grid"]:
            searcher = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["param_grid"],
                n_iter=n_iter,
                cv=cv,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
            )
            searcher.fit(X_train, y_train)
            best_model  = searcher.best_estimator_
            best_params = searcher.best_params_
            cv_auc      = searcher.best_score_
            logger.info(
                "[%s] Best CV AUC: %.4f  params: %s",
                config["name"], cv_auc, best_params,
            )
        else:
            config["model"].fit(X_train, y_train)
            best_model  = config["model"]
            best_params = {}
            cv_auc      = 0.0

        # ── Evaluate on test set ─────────────────────────────────────────
        metrics = evaluate(best_model, X_test, y_test)

        # ── Log to MLflow ────────────────────────────────────────────────
        mlflow.log_params({"model": config["name"], "tuning": use_tuning, **best_params})
        mlflow.log_metrics({**metrics, "cv_roc_auc": round(cv_auc, 4)})

        # Log model artifact
        config["log_fn"](best_model, artifact_path="model")

        # Log feature importance if available
        try:
            estimator = (
                best_model.named_steps["clf"]
                if hasattr(best_model, "named_steps")
                else best_model
            )
            if hasattr(estimator, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature":   X_train.columns,
                    "importance": estimator.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi_path = f"/tmp/{config['name']}_feature_importance.csv"
                fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path, artifact_path="feature_importance")
        except Exception:
            pass

        logger.info(
            "[%s] accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  roc_auc=%.4f",
            config["name"],
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"], metrics["roc_auc"],
        )

    return best_model, metrics, run_id


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def register_best_model(
    results: list[dict],
    registry_name: str = "credit-risk-champion",
) -> str:
    """
    Selects the run with the highest test ROC-AUC and registers it in the
    MLflow Model Registry.

    Returns the registered model version URI.
    """
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    model_uri = f"runs:/{best['run_id']}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(registry_name, "champion_model", best["name"])
    client.set_registered_model_tag(registry_name, "roc_auc",        str(best["metrics"]["roc_auc"]))

    logger.info(
        "Registered '%s' v%s as champion (ROC-AUC=%.4f)",
        registry_name, mv.version, best["metrics"]["roc_auc"],
    )
    return model_uri


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    raw_path: str        = "data/raw/data.csv",
    features_path: str   = "data/processed/features.csv",
    experiment_name: str = "credit-risk-model",
    output_dir: str      = "data/processed",
    use_tuning: bool     = True,
    n_iter: int          = 20,
    use_smote: bool      = True,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    )
    mlflow.set_experiment(experiment_name)

    # ── Data ──────────────────────────────────────────────────────────────
    features = load_or_build_features(features_path, raw_path)
    X_train, X_test, y_train, y_test = prepare_splits(
        features, use_smote=use_smote
    )

    logger.info(
        "Feature matrix: %d train, %d test, %d features",
        len(X_train), len(X_test), X_train.shape[1],
    )

    # ── Train all models ──────────────────────────────────────────────────
    results = []
    for config in get_model_configs():
        logger.info("Training %s ...", config["name"])
        model, metrics, run_id = train_and_track(
            config, X_train, y_train, X_test, y_test,
            experiment_name=experiment_name,
            use_tuning=use_tuning,
            n_iter=n_iter,
        )
        results.append({
            "name":    config["name"],
            "model":   model,
            "metrics": metrics,
            "run_id":  run_id,
        })

        # Save local copy
        joblib.dump(model, os.path.join(output_dir, f"{config['name']}.pkl"))

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["metrics"]["roc_auc"], reverse=True):
        m = r["metrics"]
        print(
            f"{r['name']:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )
    print("=" * 72)

    # ── Register champion ─────────────────────────────────────────────────
    register_best_model(results)

    # ── Also save champion as risk_model.pkl for the API ─────────────────
    champion = max(results, key=lambda r: r["metrics"]["roc_auc"])
    joblib.dump(champion["model"], os.path.join(output_dir, "risk_model.pkl"))
    logger.info("Champion model saved to %s/risk_model.pkl", output_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow tracking.")
    parser.add_argument("--data",       default="data/raw/data.csv",         help="Raw CSV path")
    parser.add_argument("--features",   default="data/processed/features.csv", help="Processed features CSV")
    parser.add_argument("--experiment", default="credit-risk-model",         help="MLflow experiment name")
    parser.add_argument("--output",     default="data/processed",            help="Output directory")
    parser.add_argument("--no-tuning",  action="store_true",                  help="Skip RandomizedSearchCV")
    parser.add_argument("--no-smote",   action="store_true",                  help="Skip SMOTE oversampling")
    parser.add_argument("--n-iter",     type=int, default=20,                 help="RandomizedSearchCV iterations")
    args = parser.parse_args()

    main(
        raw_path=args.data,
        features_path=args.features,
        experiment_name=args.experiment,
        output_dir=args.output,
        use_tuning=not args.no_tuning,
        n_iter=args.n_iter,
        use_smote=not args.no_smote,
    )
