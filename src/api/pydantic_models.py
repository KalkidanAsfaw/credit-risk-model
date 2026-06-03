from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Feature vector for one customer, matching the output of the
    Task 3/4 feature engineering pipeline (39 features).

    All WoE-encoded columns (*_woe) default to 0.0 so callers that
    only have the raw aggregates can still get a prediction.
    """

    # ── Amount aggregates ────────────────────────────────────────
    total_amount: float = Field(..., description="Total transaction amount")
    mean_amount:  float = Field(..., description="Mean transaction amount")
    std_amount:   float = Field(0.0, description="Std deviation of transaction amounts")
    max_amount:   float = Field(..., description="Maximum transaction amount")
    min_amount:   float = Field(..., description="Minimum transaction amount")

    # ── Transaction behaviour (scaled floats when coming from the pipeline) ──
    tx_count:          float = Field(..., description="Transaction count (scaled)")
    unique_products:   float = Field(..., description="Distinct products (scaled)")
    unique_categories: float = Field(..., description="Distinct categories (scaled)")
    unique_providers:  float = Field(..., description="Distinct providers (scaled)")
    unique_channels:   float = Field(..., description="Distinct channels (scaled)")
    fraud_count:       float = Field(0.0, description="Fraud transaction count (scaled)")
    fraud_rate:        float = Field(0.0, description="Fraud rate (scaled)")

    # ── Temporal features ────────────────────────────────────────
    mean_hour:   float = Field(..., description="Mean hour of day for transactions (0–23)")
    hour_std:    float = Field(0.0, description="Std deviation of transaction hour")
    mean_day:    float = Field(..., description="Mean day of month for transactions (1–31)")
    mean_month:  float = Field(..., description="Mean calendar month (1–12)")
    days_active: float = Field(..., description="Days active span (scaled)")
    tx_per_day:  float = Field(..., description="Transactions per active day")

    # ── Encoded categoricals (label-encoded mode per customer) ───
    ProductCategory_enc:  float = Field(0.0, description="Label-encoded dominant product category")
    ChannelId_enc:        float = Field(0.0, description="Label-encoded dominant channel")
    ProviderId_enc:       float = Field(0.0, description="Label-encoded dominant provider")
    PricingStrategy_enc:  float = Field(0.0, description="Label-encoded dominant pricing strategy")

    # ── WoE-encoded features (default 0.0 if not pre-computed) ──
    total_amount_woe:          float = 0.0
    mean_amount_woe:           float = 0.0
    std_amount_woe:            float = 0.0
    max_amount_woe:            float = 0.0
    min_amount_woe:            float = 0.0
    tx_count_woe:              float = 0.0
    unique_products_woe:       float = 0.0
    unique_categories_woe:     float = 0.0
    unique_providers_woe:      float = 0.0
    mean_hour_woe:             float = 0.0
    hour_std_woe:              float = 0.0
    mean_day_woe:              float = 0.0
    mean_month_woe:            float = 0.0
    days_active_woe:           float = 0.0
    tx_per_day_woe:            float = 0.0
    ProductCategory_enc_woe:   float = 0.0
    ProviderId_enc_woe:        float = 0.0

    model_config = {"json_schema_extra": {
        "example": {
            "total_amount": 1.2, "mean_amount": 0.5, "std_amount": 0.3,
            "max_amount": 1.8, "min_amount": -0.2,
            "tx_count": 12, "unique_products": 3, "unique_categories": 2,
            "unique_providers": 2, "unique_channels": 1,
            "fraud_count": 0, "fraud_rate": 0.0,
            "mean_hour": 14.0, "hour_std": 3.0, "mean_day": 15.0,
            "mean_month": 6.0, "days_active": 45, "tx_per_day": 0.27,
            "ProductCategory_enc": 0.0, "ChannelId_enc": 1.0,
            "ProviderId_enc": 2.0, "PricingStrategy_enc": 1.0,
        }
    }}


class PredictResponse(BaseModel):
    """Risk score response returned by POST /predict."""
    default_probability: float = Field(..., description="P(default) in [0, 1]")
    credit_score:        int   = Field(..., description="Credit score in [300, 850]")
    risk_category:       str   = Field(..., description="Very Low / Low / Medium / High / Very High Risk")


class HealthResponse(BaseModel):
    status: str


# ── Backward-compatible alias ────────────────────────────────────────────────
CustomerFeatures = PredictRequest
ScoreResponse    = PredictResponse
