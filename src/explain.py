"""
explain.py — SHAP-based explainability for the LightGBM champion model.

Produces two artifacts under reports/:
  - shap_global_importance.png : global mean(|SHAP|) feature ranking
  - shap_waterfall_example.png : per-prediction explanation for the
                                  highest-predicted-risk customer in the sample

Usage
-----
  python -m src.explain
"""

from __future__ import annotations

import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.train import RANDOM_STATE, TEST_SIZE, load_or_build_features, prepare_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
SAMPLE_SIZE = 200


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Exact SHAP values for a tree-based model (LightGBM/RandomForest/DecisionTree)."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def plot_global_importance(shap_values: shap.Explanation, output_path: str) -> None:
    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP importance plot to %s", output_path)


def plot_local_explanation(shap_values: shap.Explanation, index: int, output_path: str) -> None:
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved local SHAP waterfall plot (row %d) to %s", index, output_path)


def main(
    features_path: str = "data/processed/features.csv",
    raw_path: str = "data/raw/data.csv",
    model_path: str = "data/processed/risk_model.pkl",
    output_dir: str = REPORTS_DIR,
    sample_size: int = SAMPLE_SIZE,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    features = load_or_build_features(features_path, raw_path)
    _, X_test, _, _ = prepare_splits(
        features, test_size=TEST_SIZE, random_state=RANDOM_STATE, use_smote=False
    )

    model = joblib.load(model_path)
    X_test = X_test[list(model.feature_names_in_)]
    if len(X_test) > sample_size:
        X_test = X_test.sample(sample_size, random_state=RANDOM_STATE)

    shap_values = compute_shap_values(model, X_test)
    plot_global_importance(shap_values, os.path.join(output_dir, "shap_global_importance.png"))

    highest_risk_idx = int(model.predict_proba(X_test)[:, 1].argmax())
    plot_local_explanation(
        shap_values, highest_risk_idx, os.path.join(output_dir, "shap_waterfall_example.png")
    )


if __name__ == "__main__":
    main()
