import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    clean_data,
    compute_rfm,
    build_default_label,
    build_transaction_features,
    build_features,
    prepare_modelling_data,
)
from src.predict import probability_to_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    rng = np.random.default_rng(0)
    n = 200
    accounts = [f"ACC_{i:03d}" for i in range(20)]
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(n)],
        "BatchId": rng.integers(1, 10, n),
        "AccountId": rng.choice(accounts, n),
        "SubscriptionId": rng.integers(1, 5, n),
        "CurrencyCode": "ETB",
        "CountryCode": 256,
        "ProviderId": rng.integers(1, 6, n),
        "ProductId": rng.integers(1, 20, n),
        "ProductCategory": rng.choice(["airtime", "financial_services", "utility_bill"], n),
        "ChannelId": rng.integers(1, 4, n),
        "Amount": rng.uniform(10, 5000, n),
        "Value": rng.uniform(10, 5000, n),
        "TransactionStartTime": pd.date_range("2023-01-01", periods=n, freq="6h", tz="UTC"),
        "PricingStrategy": rng.integers(0, 4, n),
        "FraudResult": rng.choice([0, 1], n, p=[0.95, 0.05]),
    })


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

def test_clean_data_removes_duplicates(sample_df):
    duped = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(duped)
    assert cleaned["TransactionId"].is_unique


def test_clean_data_amounts_positive(sample_df):
    df = sample_df.copy()
    df.loc[0, "Amount"] = -100
    cleaned = clean_data(df)
    assert (cleaned["Amount"] >= 0).all()


# ---------------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------------

def test_rfm_shape(sample_df):
    df = clean_data(sample_df)
    rfm = compute_rfm(df)
    assert set(rfm.columns) == {"AccountId", "Recency", "Frequency", "Monetary"}
    assert len(rfm) == df["AccountId"].nunique()


def test_rfm_recency_non_negative(sample_df):
    df = clean_data(sample_df)
    rfm = compute_rfm(df)
    assert (rfm["Recency"] >= 0).all()


# ---------------------------------------------------------------------------
# Default label
# ---------------------------------------------------------------------------

def test_default_label_binary(sample_df):
    df = clean_data(sample_df)
    rfm = compute_rfm(df)
    labeled = build_default_label(rfm)
    assert set(labeled["is_bad"].unique()).issubset({0, 1})


def test_default_label_has_both_classes(sample_df):
    df = clean_data(sample_df)
    rfm = compute_rfm(df)
    labeled = build_default_label(rfm)
    assert labeled["is_bad"].sum() > 0
    assert (labeled["is_bad"] == 0).sum() > 0


# ---------------------------------------------------------------------------
# Transaction features
# ---------------------------------------------------------------------------

def test_transaction_features_no_nulls(sample_df):
    df = clean_data(sample_df)
    feats = build_transaction_features(df)
    assert feats.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_build_features_shape(sample_df):
    df = clean_data(sample_df)
    features = build_features(df)
    assert len(features) > 0
    assert "is_bad" in features.columns


def test_prepare_modelling_data_no_nulls(sample_df):
    df = clean_data(sample_df)
    features = build_features(df)
    X, y = prepare_modelling_data(features)
    assert X.isnull().sum().sum() == 0
    assert len(X) == len(y)


# ---------------------------------------------------------------------------
# Credit scoring
# ---------------------------------------------------------------------------

def test_score_bounds():
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        score = probability_to_score(p)
        assert 300 <= score <= 850, f"score {score} out of range for p={p}"


def test_score_monotone_decreasing():
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    scores = [probability_to_score(p) for p in probs]
    assert scores == sorted(scores, reverse=True)
