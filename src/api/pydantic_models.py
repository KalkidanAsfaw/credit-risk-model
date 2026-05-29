from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Recency: float = Field(..., description="Days since last transaction")
    Frequency: float = Field(..., description="Total number of transactions")
    Monetary: float = Field(..., description="Total transaction value")
    total_amount: float
    mean_amount: float
    std_amount: float = 0.0
    max_amount: float
    min_amount: float
    tx_count: int
    unique_products: int
    unique_categories: int
    unique_providers: int
    unique_channels: int
    fraud_count: int = 0
    fraud_rate: float = 0.0
    hour_std: float = 0.0
    days_active: int = 1
    tx_per_day: float = 0.0


class ScoreResponse(BaseModel):
    default_probability: float = Field(..., description="Probability of default (0–1)")
    credit_score: int = Field(..., description="Credit score (300–850)")
    risk_category: str
    recommended_amount: float = Field(..., description="Recommended loan amount")
    recommended_duration_months: int = Field(..., description="Recommended loan duration in months")


class HealthResponse(BaseModel):
    status: str
