"""
dashboard.py — Streamlit dashboard for the Bati Bank BNPL credit risk model.

Lets a credit officer pick a customer, see their risk score and a plain-language
lending recommendation, understand why the model scored them that way (SHAP),
and see where that customer sits in the overall portfolio.

Run
---
  streamlit run src/dashboard.py
"""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.explain import compute_shap_values
from src.predict import probability_to_score, risk_category

MODEL_PATH = os.getenv("MODEL_PATH", "data/processed/risk_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/features.csv")

RISK_COLORS = {
    "Very Low Risk": "#0ca30c",
    "Low Risk": "#5fa811",
    "Medium Risk": "#fab219",
    "High Risk": "#ec835a",
    "Very High Risk": "#d03b3b",
}
RISK_ORDER = list(RISK_COLORS.keys())

RECOMMENDATIONS = {
    "Very Low Risk": "Approve — full requested amount, standard terms.",
    "Low Risk": "Approve — standard terms.",
    "Medium Risk": "Approve with a reduced limit; consider manual review.",
    "High Risk": "Manual review required before approval.",
    "Very High Risk": "Decline.",
}

DISPLAY_COLUMNS = [
    "total_amount", "mean_amount", "tx_count", "unique_products",
    "unique_categories", "unique_providers", "days_active", "tx_per_day",
    "mean_month",
]


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_features(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def score_portfolio(_model, features: pd.DataFrame) -> pd.DataFrame:
    X = features[list(_model.feature_names_in_)]
    proba = _model.predict_proba(X)[:, 1]
    scores = probability_to_score(proba)
    categories = [risk_category(int(s)) for s in scores]
    out = features[["AccountId"]].copy()
    out["default_probability"] = proba
    out["credit_score"] = scores
    out["risk_category"] = categories
    return out


def plot_portfolio_distribution(scored: pd.DataFrame):
    counts = scored["risk_category"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [RISK_COLORS[c] for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.6, zorder=3)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}",
            ha="center", va="bottom", fontsize=10, color="#0b0b0b",
        )
    ax.set_ylabel("Customers")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def render_customer_panel(model, features: pd.DataFrame, scored: pd.DataFrame, account_id: str) -> None:
    row = features[features["AccountId"] == account_id].iloc[0]
    result = scored[scored["AccountId"] == account_id].iloc[0]
    category = result["risk_category"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{result['default_probability']:.1%}")
    col2.metric("Credit Score", int(result["credit_score"]))
    col3.markdown(
        f"<div style='padding:10px;border-radius:6px;background:{RISK_COLORS[category]}22;"
        f"border:1px solid {RISK_COLORS[category]};text-align:center;margin-top:4px'>"
        f"<b>{category}</b></div>",
        unsafe_allow_html=True,
    )

    st.info(f"**Recommended action:** {RECOMMENDATIONS[category]}")

    st.markdown("**Customer behavior snapshot**")
    st.dataframe(row[DISPLAY_COLUMNS].astype(float).round(2).to_frame("value"))

    st.markdown("**Why this score? (SHAP)**")
    X_customer = row[list(model.feature_names_in_)].to_frame().T.astype(float)
    shap_values = compute_shap_values(model, X_customer)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    plt.close("all")


def main() -> None:
    st.set_page_config(page_title="Bati Bank — Credit Risk Scoring", page_icon="💳", layout="wide")
    st.title("Bati Bank BNPL Credit Risk Scoring")
    st.caption(
        "Turns eCommerce shopping behavior into a real-time default probability, "
        "credit score, and lending recommendation for customers with no prior loan history."
    )

    model = load_model(MODEL_PATH)
    features = load_features(FEATURES_PATH)

    if model is None or features is None:
        st.error(
            "Model or feature data not found. Run `python -m src.train` first to "
            f"produce `{MODEL_PATH}` and `{FEATURES_PATH}`."
        )
        return

    scored = score_portfolio(model, features)

    st.subheader("Portfolio risk distribution")
    st.pyplot(plot_portfolio_distribution(scored))
    st.caption(
        "A large share of the current portfolio falls into Very High Risk — this is a direct "
        "consequence of how the is_high_risk proxy label was constructed from RFM/K-Means "
        "clustering (see the README's proxy-variable risk disclosure), not a confirmed default rate."
    )

    st.subheader("Score an individual customer")
    account_ids = sorted(features["AccountId"].unique())
    selected = st.selectbox("Customer (AccountId)", account_ids)
    render_customer_panel(model, features, scored, selected)


if __name__ == "__main__":
    main()
