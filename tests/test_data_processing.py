import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    clean_data,
    compute_rfm,
    compute_customer_rfm,
    build_default_label,
    assign_high_risk_label,
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
    accounts  = [f"ACC_{i:03d}"  for i in range(20)]
    customers = [f"CUST_{i:03d}" for i in range(22)]   # slightly more customers than accounts

    acc_arr  = rng.choice(accounts, n)
    # each account maps to a stable customer (1 or 2 customers per account)
    acc_to_cust = {a: rng.choice(customers, 1)[0] for a in accounts}
    cust_arr = np.array([acc_to_cust[a] for a in acc_arr])

    return pd.DataFrame({
        "TransactionId":       [f"T{i}" for i in range(n)],
        "BatchId":             rng.integers(1, 10, n),
        "AccountId":           acc_arr,
        "CustomerId":          cust_arr,
        "SubscriptionId":      rng.integers(1, 5, n),
        "CurrencyCode":        "ETB",
        "CountryCode":         256,
        "ProviderId":          rng.integers(1, 6, n),
        "ProductId":           rng.integers(1, 20, n),
        "ProductCategory":     rng.choice(["airtime", "financial_services", "utility_bill"], n),
        "ChannelId":           rng.integers(1, 4, n),
        "Amount":              rng.uniform(10, 5000, n),
        "Value":               rng.uniform(10, 5000, n),
        "TransactionStartTime": pd.date_range("2023-01-01", periods=n, freq="6h", tz="UTC"),
        "PricingStrategy":     rng.integers(0, 4, n),
        "FraudResult":         rng.choice([0, 1], n, p=[0.95, 0.05]),
    })


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

def test_clean_data_removes_duplicates(sample_df):
    duped   = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(duped)
    assert cleaned["TransactionId"].is_unique


def test_clean_data_amounts_positive(sample_df):
    df = sample_df.copy()
    df.loc[0, "Amount"] = -100
    cleaned = clean_data(df)
    assert (cleaned["Amount"] >= 0).all()


# ---------------------------------------------------------------------------
# Legacy RFM (AccountId)
# ---------------------------------------------------------------------------

def test_rfm_shape(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_rfm(df)
    assert set(rfm.columns) == {"AccountId", "Recency", "Frequency", "Monetary"}
    assert len(rfm) == df["AccountId"].nunique()


def test_rfm_recency_non_negative(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_rfm(df)
    assert (rfm["Recency"] >= 0).all()


# ---------------------------------------------------------------------------
# Task 4 – Customer RFM (CustomerId)
# ---------------------------------------------------------------------------

def test_customer_rfm_shape(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_customer_rfm(df)
    assert set(rfm.columns) == {"CustomerId", "Recency", "Frequency", "Monetary"}
    assert len(rfm) == df["CustomerId"].nunique()


def test_customer_rfm_recency_non_negative(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_customer_rfm(df)
    assert (rfm["Recency"] >= 0).all()


def test_customer_rfm_frequency_positive(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_customer_rfm(df)
    assert (rfm["Frequency"] > 0).all()


def test_customer_rfm_snapshot_date(sample_df):
    df            = clean_data(sample_df)
    snapshot_date = pd.Timestamp("2030-01-01", tz="UTC")
    rfm           = compute_customer_rfm(df, snapshot_date=snapshot_date)
    # All recency values must be positive with a far-future snapshot
    assert (rfm["Recency"] > 0).all()


# ---------------------------------------------------------------------------
# Task 4 – High-risk label (is_high_risk)
# ---------------------------------------------------------------------------

def test_high_risk_label_binary(sample_df):
    df      = clean_data(sample_df)
    rfm     = compute_customer_rfm(df)
    labeled = assign_high_risk_label(rfm)
    assert set(labeled["is_high_risk"].unique()).issubset({0, 1})


def test_high_risk_label_has_both_classes(sample_df):
    df      = clean_data(sample_df)
    rfm     = compute_customer_rfm(df)
    labeled = assign_high_risk_label(rfm)
    assert labeled["is_high_risk"].sum() > 0
    assert (labeled["is_high_risk"] == 0).sum() > 0


def test_high_risk_label_columns(sample_df):
    df      = clean_data(sample_df)
    rfm     = compute_customer_rfm(df)
    labeled = assign_high_risk_label(rfm)
    required = {"CustomerId", "Recency", "Frequency", "Monetary",
                "cluster", "is_high_risk"}
    assert required.issubset(set(labeled.columns))


def test_high_risk_label_reproducible(sample_df):
    df  = clean_data(sample_df)
    rfm = compute_customer_rfm(df)
    l1  = assign_high_risk_label(rfm, random_state=42)
    l2  = assign_high_risk_label(rfm, random_state=42)
    pd.testing.assert_frame_equal(l1.reset_index(drop=True),
                                  l2.reset_index(drop=True))


def test_high_risk_cluster_is_least_engaged(sample_df):
    """High-risk cluster must have the highest mean Recency among all clusters."""
    df      = clean_data(sample_df)
    rfm     = compute_customer_rfm(df)
    labeled = assign_high_risk_label(rfm)

    profiles = labeled.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = labeled.loc[
        labeled["is_high_risk"] == 1, "cluster"
    ].iloc[0]

    assert profiles.loc[high_risk_cluster, "Recency"] == profiles["Recency"].max()


# ---------------------------------------------------------------------------
# Legacy default label (AccountId / is_bad)
# ---------------------------------------------------------------------------

def test_default_label_binary(sample_df):
    df      = clean_data(sample_df)
    rfm     = compute_rfm(df)
    labeled = build_default_label(rfm)
    assert set(labeled["is_bad"].unique()).issubset({0, 1})


def test_default_label_has_both_classes(sample_df):
    df      = clean_data(sample_df)
    rfm     = compute_rfm(df)
    labeled = build_default_label(rfm)
    assert labeled["is_bad"].sum() > 0
    assert (labeled["is_bad"] == 0).sum() > 0


# ---------------------------------------------------------------------------
# Transaction features
# ---------------------------------------------------------------------------

def test_transaction_features_no_nulls(sample_df):
    df    = clean_data(sample_df)
    feats = build_transaction_features(df)
    assert feats.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Full pipeline (legacy)
# ---------------------------------------------------------------------------

def test_build_features_shape(sample_df):
    df       = clean_data(sample_df)
    features = build_features(df)
    assert len(features) > 0
    assert "is_bad" in features.columns


def test_prepare_modelling_data_no_nulls(sample_df):
    df       = clean_data(sample_df)
    features = build_features(df)
    X, y     = prepare_modelling_data(features)
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
    probs  = [0.1, 0.3, 0.5, 0.7, 0.9]
    scores = [probability_to_score(p) for p in probs]
    assert scores == sorted(scores, reverse=True)
