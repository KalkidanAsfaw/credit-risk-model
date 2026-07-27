"""
Scoring utilities:
  - probability_to_score : maps P(default) → credit score
  - risk_category         : maps a credit score to a human-readable risk band
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreConfig:
    """Scorecard scaling and risk-band configuration (FICO-style, log-odds linear transform)."""

    score_min: int = 300
    score_max: int = 850
    pdo: int = 20                  # points-to-double-odds
    base_score: int = 600
    base_odds: int = 50            # good:bad odds at base score
    risk_bands: tuple[tuple[int, str], ...] = (
        (750, "Very Low Risk"),
        (670, "Low Risk"),
        (580, "Medium Risk"),
        (500, "High Risk"),
    )


DEFAULT_SCORE_CONFIG = ScoreConfig()


def probability_to_score(
    probability: float | np.ndarray,
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> float | np.ndarray:
    """Converts P(default) to a credit score in [config.score_min, config.score_max]."""
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    odds = (1 - probability) / probability
    factor = config.pdo / np.log(2)
    offset = config.base_score - factor * np.log(config.base_odds)
    score = offset + factor * np.log(odds)
    return np.clip(score, config.score_min, config.score_max).round().astype(int)


def risk_category(score: int, config: ScoreConfig = DEFAULT_SCORE_CONFIG) -> str:
    """Maps a credit score to a human-readable risk band."""
    for threshold, label in config.risk_bands:
        if score >= threshold:
            return label
    return "Very High Risk"
