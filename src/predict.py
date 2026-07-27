"""
Scoring utilities:
  - probability_to_score : maps P(default) → credit score (300–850)
  - risk_category         : maps a credit score to a human-readable risk band
"""

import numpy as np


# ---------------------------------------------------------------------------
# Credit score scaling (log-odds linear transform, FICO-style)
# ---------------------------------------------------------------------------

SCORE_MIN = 300
SCORE_MAX = 850
PDO = 20          # points-to-double-odds
BASE_SCORE = 600
BASE_ODDS = 50    # good:bad odds at base score


def probability_to_score(probability: float | np.ndarray) -> float | np.ndarray:
    """Converts P(default) to a credit score in [300, 850]."""
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    odds = (1 - probability) / probability
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    score = offset + factor * np.log(odds)
    return np.clip(score, SCORE_MIN, SCORE_MAX).round().astype(int)


def risk_category(score: int) -> str:
    """Maps a credit score to a human-readable risk band."""
    if score >= 750:
        return "Very Low Risk"
    if score >= 670:
        return "Low Risk"
    if score >= 580:
        return "Medium Risk"
    if score >= 500:
        return "High Risk"
    return "Very High Risk"
