"""
data_processing.py

Full feature engineering pipeline for credit risk modelling.

Pipeline architecture (transaction-level → customer-level):
  Raw DataFrame
      │
  1.  DatetimeFeatureExtractor   – adds tx_hour, tx_day, tx_month, tx_year
      │
  2.  CustomerFeatureAggregator  – groups by AccountId; computes aggregate,
      │                            datetime, behavioural, and encoded features
      │
  3.  NumericalPreprocessor      – median imputation + StandardScaler
      │
  (4. WoEEncoder)                – Weight of Evidence, fitted separately with
                                   the RFM-derived target (is_bad)

Entry points
------------
  build_pipeline()          – returns an unfitted sklearn Pipeline (steps 1–3)
  fit_full_pipeline(df)     – cleans → pipeline → RFM label → WoE → saves CSV
"""

from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# ── Feature column lists ──────────────────────────────────────────────────────

_NUMERIC_FEATURES = [
    "total_amount", "mean_amount", "std_amount", "max_amount", "min_amount",
    "tx_count", "unique_products", "unique_categories",
    "unique_providers", "unique_channels",
    "fraud_count", "fraud_rate",
    "mean_hour", "hour_std", "mean_day", "mean_month",
    "days_active", "tx_per_day",
    "Recency", "Frequency", "Monetary",
    "ProductCategory_enc", "ChannelId_enc", "ProviderId_enc", "PricingStrategy_enc",
]

_CATEGORICAL_COLS = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"]


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers  (kept for backward-compatibility with existing unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["TransactionStartTime"])


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


def compute_rfm(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
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


def build_default_label(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    rfm = rfm.copy()
    features = ["Recency", "Frequency", "Monetary"]
    X = StandardScaler().fit_transform(rfm[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X)
    summary = rfm.groupby("cluster")[features].mean()
    summary["risk_score"] = (
        summary["Recency"].rank(ascending=True)      # high Recency = inactive = risky
        + summary["Frequency"].rank(ascending=False)  # low Frequency = infrequent = risky
        + summary["Monetary"].rank(ascending=False)  # low Monetary = low spend = risky
    )
    bad_cluster = int(summary["risk_score"].idxmax())
    rfm["is_bad"] = (rfm["cluster"] == bad_cluster).astype(int)
    return rfm[["AccountId", "Recency", "Frequency", "Monetary", "is_bad"]]


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

    df2 = df.copy()
    df2["hour"] = df2["TransactionStartTime"].dt.hour
    hour_std = df2.groupby("AccountId")["hour"].std().reset_index(name="hour_std")
    agg = agg.merge(hour_std, on="AccountId", how="left")
    agg["hour_std"] = agg["hour_std"].fillna(0)

    span = (
        df.groupby("AccountId")["TransactionStartTime"]
        .agg(lambda x: (x.max() - x.min()).days + 1)
        .reset_index(name="days_active")
    )
    agg = agg.merge(span, on="AccountId", how="left")
    agg["tx_per_day"] = agg["tx_count"] / agg["days_active"].clip(lower=1)
    return agg


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rfm_labeled = build_default_label(compute_rfm(df))
    tx_feats = build_transaction_features(df)
    return rfm_labeled.merge(tx_feats, on="AccountId", how="inner")


def get_feature_columns() -> list[str]:
    return [
        "Recency", "Frequency", "Monetary",
        "total_amount", "mean_amount", "std_amount", "max_amount", "min_amount",
        "tx_count", "unique_products", "unique_categories",
        "unique_providers", "unique_channels",
        "fraud_count", "fraud_rate",
        "hour_std", "days_active", "tx_per_day",
    ]


def prepare_modelling_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # prefer is_high_risk (Task 4) over legacy is_bad
    target_col = "is_high_risk" if "is_high_risk" in features.columns else "is_bad"
    y = features[target_col]

    # Use hardcoded feature list if all columns are present (legacy pipeline),
    # otherwise fall back to all numeric columns except IDs and target
    hardcoded = get_feature_columns()
    if all(c in features.columns for c in hardcoded):
        X = features[hardcoded].fillna(0)
    else:
        exclude = {"AccountId", "CustomerId", "is_high_risk", "is_bad", "cluster"}
        X = features.select_dtypes(include=[np.number]).drop(
            columns=[c for c in exclude if c in features.columns],
            errors="ignore",
        ).fillna(0)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 – Proxy Target Variable (is_high_risk)
# ─────────────────────────────────────────────────────────────────────────────

def compute_customer_rfm(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Computes RFM metrics per CustomerId.

    Recency  – days since the customer's most recent transaction
               (lower = more engaged)
    Frequency – total number of transactions
               (higher = more engaged)
    Monetary  – total spend (sum of Amount)
               (higher = more engaged)

    Parameters
    ----------
    df            : cleaned transaction DataFrame (must contain CustomerId,
                    TransactionStartTime, TransactionId, Amount)
    snapshot_date : reference date for Recency; defaults to max date + 1 day

    Returns
    -------
    DataFrame with columns: CustomerId, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    return rfm


def assign_high_risk_label(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Segments customers into 3 clusters using K-Means on scaled RFM features
    and assigns is_high_risk=1 to the most disengaged cluster.

    High-risk cluster identification
    ---------------------------------
    The cluster with the combination of:
      • highest Recency  (customer has been inactive the longest)
      • lowest Frequency (customer transacts least often)
      • lowest Monetary  (customer spends the least)
    receives is_high_risk=1.  All other customers receive 0.

    Parameters
    ----------
    rfm          : DataFrame returned by compute_customer_rfm()
    n_clusters   : number of K-Means clusters (default 3)
    random_state : seed for reproducibility (default 42)

    Returns
    -------
    DataFrame with columns:
        CustomerId, Recency, Frequency, Monetary, cluster, is_high_risk
    """
    rfm = rfm.copy()
    features = ["Recency", "Frequency", "Monetary"]

    # Scale before clustering so no single dimension dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    # Rank each cluster: high Recency + low Frequency + low Monetary = high risk
    cluster_profiles = rfm.groupby("cluster")[features].mean()
    cluster_profiles["risk_score"] = (
        cluster_profiles["Recency"].rank(ascending=True)      # high Recency = inactive = risky
        + cluster_profiles["Frequency"].rank(ascending=False) # low Frequency = infrequent = risky
        + cluster_profiles["Monetary"].rank(ascending=False)  # low Monetary = low spend = risky
    )

    high_risk_cluster = int(cluster_profiles["risk_score"].idxmax())
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    logger.info(
        "assign_high_risk_label: cluster %d identified as high-risk "
        "(Recency=%.1f, Frequency=%.1f, Monetary=%.0f). "
        "%d / %d customers labelled is_high_risk=1 (%.1f%%)",
        high_risk_cluster,
        cluster_profiles.loc[high_risk_cluster, "Recency"],
        cluster_profiles.loc[high_risk_cluster, "Frequency"],
        cluster_profiles.loc[high_risk_cluster, "Monetary"],
        rfm["is_high_risk"].sum(),
        len(rfm),
        rfm["is_high_risk"].mean() * 100,
    )

    return rfm[["CustomerId", "Recency", "Frequency", "Monetary",
                "cluster", "is_high_risk"]]


# ─────────────────────────────────────────────────────────────────────────────
# Custom sklearn Transformers
# ─────────────────────────────────────────────────────────────────────────────

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from TransactionStartTime.

    Adds four columns (transaction-level):
        tx_hour   – hour of day  (0–23)
        tx_day    – day of month (1–31)
        tx_month  – month        (1–12)
        tx_year   – calendar year
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DatetimeFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ts = pd.to_datetime(X["TransactionStartTime"], utc=True, errors="coerce")
        X["tx_hour"]  = ts.dt.hour
        X["tx_day"]   = ts.dt.day
        X["tx_month"] = ts.dt.month
        X["tx_year"]  = ts.dt.year
        return X


class CustomerFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level rows into one row per customer (AccountId).

    Features produced
    -----------------
    Aggregate amount  : total_amount, mean_amount, std_amount, max_amount, min_amount
    Transaction count : tx_count
    Cardinality       : unique_products, unique_categories, unique_providers,
                        unique_channels
    Fraud behaviour   : fraud_count, fraud_rate
    Temporal          : mean_hour, hour_std, mean_day, mean_month
    Activity span     : days_active, tx_per_day
    Categorical modes : ProductCategory_enc, ChannelId_enc, ProviderId_enc,
                        PricingStrategy_enc  (label-encoded most-frequent value)
    """

    def __init__(self) -> None:
        self._label_encoders: dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CustomerFeatureAggregator":
        # Fit label encoders on the full transaction-level data
        for col in _CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str).fillna("__missing__"))
                self._label_encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"], utc=True, errors="coerce"
        )

        # ── Core amount aggregates ─────────────────────────────────────────
        agg = X.groupby("AccountId").agg(
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

        # ── Datetime aggregates (uses extracted columns when available) ─────
        time_cols: dict[str, tuple[str, str]] = {}
        if "tx_hour" in X.columns:
            time_cols["mean_hour"]  = ("tx_hour",  "mean")
            time_cols["hour_std"]   = ("tx_hour",  "std")
            time_cols["mean_day"]   = ("tx_day",   "mean")
            time_cols["mean_month"] = ("tx_month", "mean")

        if time_cols:
            time_agg = X.groupby("AccountId").agg(**{
                k: (v[0], v[1]) for k, v in time_cols.items()
            }).reset_index()
            time_agg["hour_std"] = time_agg["hour_std"].fillna(0)
            agg = agg.merge(time_agg, on="AccountId", how="left")
        else:
            # Fallback: compute from TransactionStartTime directly
            df2 = X.copy()
            df2["_hour"] = df2["TransactionStartTime"].dt.hour
            df2["_day"]  = df2["TransactionStartTime"].dt.day
            df2["_mon"]  = df2["TransactionStartTime"].dt.month
            t2 = df2.groupby("AccountId").agg(
                mean_hour=("_hour", "mean"),
                hour_std=("_hour", "std"),
                mean_day=("_day", "mean"),
                mean_month=("_mon", "mean"),
            ).reset_index()
            t2["hour_std"] = t2["hour_std"].fillna(0)
            agg = agg.merge(t2, on="AccountId", how="left")

        # ── Activity span ─────────────────────────────────────────────────
        span = (
            X.groupby("AccountId")["TransactionStartTime"]
            .agg(lambda x: (x.max() - x.min()).days + 1)
            .reset_index(name="days_active")
        )
        agg = agg.merge(span, on="AccountId", how="left")
        agg["tx_per_day"] = agg["tx_count"] / agg["days_active"].clip(lower=1)

        # ── Categorical encoding (mode per customer, label-encoded) ────────
        for col in _CATEGORICAL_COLS:
            if col not in X.columns:
                continue
            mode_col = (
                X.groupby("AccountId")[col]
                .agg(lambda s: s.mode().iat[0] if len(s) > 0 else np.nan)
                .reset_index()
            )
            mode_col[col] = mode_col[col].astype(str).fillna("__missing__")
            if col in self._label_encoders:
                le = self._label_encoders[col]
                # handle unseen labels gracefully
                known = set(le.classes_)
                mode_col[col] = mode_col[col].apply(
                    lambda v: v if v in known else "__missing__"
                )
                mode_col[f"{col}_enc"] = le.transform(mode_col[col])
            else:
                le = LabelEncoder()
                mode_col[f"{col}_enc"] = le.fit_transform(mode_col[col])
            agg = agg.merge(
                mode_col[["AccountId", f"{col}_enc"]], on="AccountId", how="left"
            )

        return agg


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Applies imputation followed by feature scaling to all numeric columns.

    Parameters
    ----------
    impute_strategy : str
        'median' (default), 'mean', or 'most_frequent'
    scaling : str
        'standard' (default) → StandardScaler  |  'minmax' → MinMaxScaler
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        scaling: str = "standard",
    ) -> None:
        self.impute_strategy = impute_strategy
        self.scaling = scaling
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | MinMaxScaler | None = None
        self._numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericalPreprocessor":
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self._imputer = SimpleImputer(strategy=self.impute_strategy)
        imputed = self._imputer.fit_transform(X[self._numeric_cols])

        self._scaler = (
            MinMaxScaler() if self.scaling == "minmax" else StandardScaler()
        )
        self._scaler.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cols = [c for c in self._numeric_cols if c in X.columns]
        imputed = self._imputer.transform(X[cols])
        scaled  = self._scaler.transform(imputed)
        X[cols] = scaled
        return X


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) encoder for numeric features.

    For each numeric feature the data is quantile-binned and each bin is
    replaced by its WoE score:
        WoE_i = ln( P(X=i | Y=1) / P(X=i | Y=0) )

    Features are ranked by Information Value (IV):
        IV = Σ_i (P(X=i|Y=1) − P(X=i|Y=0)) × WoE_i

    IV interpretation
    -----------------
    < 0.02  → Useless       0.02–0.1  → Weak
    0.1–0.3 → Medium        0.3–0.5   → Strong     > 0.5 → Very Strong

    Parameters
    ----------
    n_bins       : number of quantile bins (default 10)
    iv_threshold : minimum IV to retain a feature's WoE column (default 0.02)

    Usage
    -----
    Requires binary y at fit time.  Usually applied after build_features() so
    that the RFM-derived is_bad target is available.
    """

    def __init__(self, n_bins: int = 10, iv_threshold: float = 0.02) -> None:
        self.n_bins = n_bins
        self.iv_threshold = iv_threshold
        self._bin_edges: dict[str, np.ndarray] = {}
        self._woe_maps:  dict[str, dict]       = {}
        self._iv_values: dict[str, float]      = {}
        self.selected_features_: list[str]     = []

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True).astype(int)

        total_events    = float(y.sum())
        total_nonevents = float((1 - y).sum())

        if total_events == 0 or total_nonevents == 0:
            logger.warning("WoEEncoder: target has only one class; skipping fit.")
            return self

        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col].fillna(X[col].median())

            # Quantile binning; fall back to equal-width if too many ties
            try:
                _, edges = pd.qcut(
                    series, q=self.n_bins, retbins=True, duplicates="drop"
                )
            except Exception:
                _, edges = pd.cut(series, bins=self.n_bins, retbins=True)

            # Extend edges to cover the full value range
            edges[0]  = -np.inf
            edges[-1] =  np.inf

            binned = pd.cut(series, bins=edges, include_lowest=True)
            tmp    = pd.DataFrame({"bin": binned, "target": y})

            stats = (
                tmp.groupby("bin", observed=True)["target"]
                .agg(events="sum", count="count")
                .reset_index()
            )
            stats["nonevents"] = stats["count"] - stats["events"]

            eps = 1e-6
            stats["pct_events"]    = (stats["events"]    / total_events).clip(lower=eps)
            stats["pct_nonevents"] = (stats["nonevents"] / total_nonevents).clip(lower=eps)
            stats["woe"] = np.log(stats["pct_events"] / stats["pct_nonevents"])
            stats["iv"]  = (stats["pct_events"] - stats["pct_nonevents"]) * stats["woe"]

            iv = float(stats["iv"].sum())
            self._iv_values[col] = iv
            self._bin_edges[col] = edges
            self._woe_maps[col]  = dict(zip(stats["bin"], stats["woe"]))

            if iv >= self.iv_threshold:
                self.selected_features_.append(col)

        logger.info(
            "WoEEncoder: %d/%d features selected (IV >= %.3f)",
            len(self.selected_features_),
            len(self._iv_values),
            self.iv_threshold,
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()

        for col in self.selected_features_:
            if col not in X.columns:
                continue
            series = X[col].fillna(X[col].median())
            binned = pd.cut(
                series,
                bins=self._bin_edges[col],
                include_lowest=True,
            )
            X[f"{col}_woe"] = binned.map(self._woe_maps[col]).astype(float).fillna(0.0)

        return X

    # ------------------------------------------------------------------
    def iv_report(self) -> pd.DataFrame:
        """Returns a DataFrame of all feature IVs sorted descending."""
        return (
            pd.DataFrame.from_dict(
                self._iv_values, orient="index", columns=["IV"]
            )
            .sort_values("IV", ascending=False)
            .assign(
                selected=lambda d: d.index.isin(self.selected_features_),
                strength=lambda d: d["IV"].map(_iv_strength),
            )
        )


def _iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Useless"
    if iv < 0.10:
        return "Weak"
    if iv < 0.30:
        return "Medium"
    if iv < 0.50:
        return "Strong"
    return "Very Strong"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(scaling: str = "standard") -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline (steps 1–3).

    Steps
    -----
    datetime_extractor  – DatetimeFeatureExtractor
    aggregator          – CustomerFeatureAggregator
    preprocessor        – NumericalPreprocessor (impute + scale)

    WoE encoding is handled separately by fit_full_pipeline() because it
    requires the RFM-derived binary target which is computed after aggregation.
    """
    return Pipeline([
        ("datetime_extractor", DatetimeFeatureExtractor()),
        ("aggregator",         CustomerFeatureAggregator()),
        ("preprocessor",       NumericalPreprocessor(scaling=scaling)),
    ])


def fit_full_pipeline(
    df: pd.DataFrame,
    output_path: str | None = None,
    scaling: str = "standard",
    use_woe: bool = True,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    End-to-end pipeline from raw transactions to a model-ready DataFrame.

    Steps
    -----
    1. clean_data           – dedup, abs(Amount), parse timestamps
    2. build_pipeline()     – datetime extract → aggregate → impute + scale
    3. build_default_label  – RFM K-Means → is_bad proxy target
    4. WoEEncoder (optional)– fit on (X, is_bad); append *_woe columns

    Parameters
    ----------
    df          : raw transaction DataFrame
    output_path : if provided, saves the processed DataFrame as CSV
    scaling     : 'standard' or 'minmax'
    use_woe     : whether to fit and apply WoE encoding
    random_state: passed to KMeans

    Returns
    -------
    (fitted_pipeline, processed_dataframe)
    """
    df_clean = clean_data(df)

    # ── Steps 1–3: transaction → account-level feature matrix ─────────────
    pipe = build_pipeline(scaling=scaling)
    customer_df = pipe.fit_transform(df_clean)

    # ── Task 4: RFM proxy label (per CustomerId) ───────────────────────────
    #
    # Recency / Frequency / Monetary are computed at the CustomerId level
    # because one customer may hold several AccountIds.  is_high_risk is then
    # mapped back to AccountId level via the customer→account bridge:
    #   • if any CustomerId linked to an AccountId is high-risk, the account
    #     is flagged (max aggregation).

    rfm_customer = compute_customer_rfm(df_clean)
    labeled      = assign_high_risk_label(rfm_customer, random_state=random_state)

    # Save the customer-level RFM + label for inspection / audit
    rfm_output_path = (
        os.path.join(os.path.dirname(output_path), "rfm_labeled.csv")
        if output_path else None
    )
    if rfm_output_path:
        out_dir = os.path.dirname(rfm_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        labeled.to_csv(rfm_output_path, index=False)
        logger.info("RFM labels saved to %s", rfm_output_path)

    # Bridge: CustomerId → AccountId  (take most frequent AccountId per customer)
    cust_to_acct = (
        df_clean.groupby("CustomerId")["AccountId"]
        .agg(lambda s: s.mode().iat[0])
        .reset_index(name="AccountId")
    )
    high_risk_per_account = (
        labeled[["CustomerId", "is_high_risk"]]
        .merge(cust_to_acct, on="CustomerId", how="left")
        .groupby("AccountId")["is_high_risk"]
        .max()                     # flag account if any linked customer is high-risk
        .reset_index()
    )

    customer_df = customer_df.merge(
        high_risk_per_account, on="AccountId", how="left"
    )
    customer_df["is_high_risk"] = customer_df["is_high_risk"].fillna(0).astype(int)

    # ── WoE encoding (uses is_high_risk as target) ─────────────────────────
    if use_woe:
        non_feature = {"AccountId", "is_high_risk", "is_bad"}
        feature_cols = [
            c for c in customer_df.columns
            if c not in non_feature
            and pd.api.types.is_numeric_dtype(customer_df[c])
        ]
        X_num = customer_df[feature_cols]
        y     = customer_df["is_high_risk"]

        woe = WoEEncoder(n_bins=10, iv_threshold=0.02)
        woe.fit(X_num, y)
        woe_out  = woe.transform(X_num)
        woe_cols = [c for c in woe_out.columns if c.endswith("_woe")]

        if woe_cols:
            customer_df = pd.concat(
                [customer_df, woe_out[woe_cols].reset_index(drop=True)],
                axis=1,
            )
            logger.info("WoE columns added: %s", woe_cols)

        pipe.woe_encoder_ = woe
        pipe.iv_report_   = woe.iv_report()

    # ── Persist processed dataset ─────────────────────────────────────────
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        customer_df.to_csv(output_path, index=False)
        logger.info("Processed dataset saved to %s (%s rows × %s cols)",
                    output_path, *customer_df.shape)

    return pipe, customer_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the feature engineering pipeline.")
    parser.add_argument("--input",  default="data/raw/data.csv",            help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed/features.csv",  help="Output CSV path")
    parser.add_argument("--scaling", default="standard", choices=["standard", "minmax"])
    parser.add_argument("--no-woe", action="store_true", help="Skip WoE encoding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = load_data(args.input)
    _, processed = fit_full_pipeline(
        raw,
        output_path=args.output,
        scaling=args.scaling,
        use_woe=not args.no_woe,
    )
    print(f"Done. Output shape: {processed.shape}")
    print(processed.head(3).to_string())
