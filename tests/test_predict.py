import pytest

from src.predict import risk_category


@pytest.mark.parametrize(
    "score,expected",
    [
        (800, "Very Low Risk"),
        (750, "Very Low Risk"),
        (700, "Low Risk"),
        (670, "Low Risk"),
        (600, "Medium Risk"),
        (580, "Medium Risk"),
        (550, "High Risk"),
        (500, "High Risk"),
        (400, "Very High Risk"),
        (300, "Very High Risk"),
    ],
)
def test_risk_category_bands(score, expected):
    assert risk_category(score) == expected
