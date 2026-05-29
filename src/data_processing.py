"""
Data processing pipeline: loads raw eCommerce transactions, engineers features,
and constructs a proxy default label via RFM-based clustering.

Expected raw CSV columns (Xente/similar format):
    TransactionId, BatchId, AccountId, SubscriptionId, CurrencyCode,
    CountryCode, ProviderId, ProductId, ProductCategory, ChannelId,
    Amount, Value, TransactionStartTime, PricingStrategy, FraudResult
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["TransactionId"])
    df["Amount"] = df["Amount"].abs()
    df["Value"] = df["Value"].abs()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["TransactionStartTime", "AccountId", "Amount"])
    return df


# ---------------------------------------------------------------------------
# RFM feature construction (per account)
# ---------------------------------------------------------------------------

def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("AccountId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    return rfm


# ---------------------------------------------------------------------------
# Proxy default label via RFM clustering
# ---------------------------------------------------------------------------

def build_default_label(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Clusters customers on RFM scores.  The cluster with the highest Recency
    (least recent), lowest Frequency and lowest Monetary value is labeled
    is_bad=1 (high-risk / proxy default).
    """
    rfm = rfm.copy()
    features = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X)

    cluster_summary = rfm.groupby("cluster")[features].mean()
    # Bad cluster: highest recency, lowest frequency, lowest monetary
    cluster_summary["risk_score"] = (
        cluster_summary["Recency"].rank(ascending=False)
        + cluster_summary["Frequency"].rank(ascending=True)
        + cluster_summary["Monetary"].rank(ascending=True)
    )
    bad_cluster = cluster_summary["risk_score"].idxmax()

    rfm["is_bad"] = (rfm["cluster"] == bad_cluster).astype(int)
    return rfm[["AccountId", "Recency", "Frequency", "Monetary", "is_bad"]]


# ---------------------------------------------------------------------------
# Aggregate transaction features (per account)
# ---------------------------------------------------------------------------

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("AccountId").agg(
        total_amount=("Amount", "sum"),
        mean_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        max_amount=("Amount", "max"),
        min_amount=("Amount", "min"),
        tx_count=("TransactionId", "count"),
        unique_products=("ProductId", "nunique"),
        unique_categories=("ProductCategory", "nunique"),
        unique_providers=("ProviderId", "nunique"),
        unique_channels=("ChannelId", "nunique"),
        fraud_count=("FraudResult", "sum"),
        fraud_rate=("FraudResult", "mean"),
    ).reset_index()

    agg["std_amount"] = agg["std_amount"].fillna(0)

    # Hour-of-day spread
    df2 = df.copy()
    df2["hour"] = df2["TransactionStartTime"].dt.hour
    hour_std = df2.groupby("AccountId")["hour"].std().reset_index(name="hour_std")
    agg = agg.merge(hour_std, on="AccountId", how="left")
    agg["hour_std"] = agg["hour_std"].fillna(0)

    # Days active
    span = (
        df.groupby("AccountId")["TransactionStartTime"]
        .agg(lambda x: (x.max() - x.min()).days + 1)
        .reset_index(name="days_active")
    )
    agg = agg.merge(span, on="AccountId", how="left")
    agg["tx_per_day"] = agg["tx_count"] / agg["days_active"].clip(lower=1)

    return agg


# ---------------------------------------------------------------------------
# Master feature table
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rfm_labeled = build_default_label(compute_rfm(df))
    tx_feats = build_transaction_features(df)
    features = rfm_labeled.merge(tx_feats, on="AccountId", how="inner")
    return features


# ---------------------------------------------------------------------------
# Train/test split helpers
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    return [
        "Recency", "Frequency", "Monetary",
        "total_amount", "mean_amount", "std_amount", "max_amount", "min_amount",
        "tx_count", "unique_products", "unique_categories",
        "unique_providers", "unique_channels",
        "fraud_count", "fraud_rate",
        "hour_std", "days_active", "tx_per_day",
    ]


def prepare_modelling_data(features: pd.DataFrame):
    X = features[get_feature_columns()].fillna(0)
    y = features["is_bad"]
    return X, y
