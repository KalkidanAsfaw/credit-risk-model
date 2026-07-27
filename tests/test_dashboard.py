import os

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = "src/dashboard.py"

requires_local_artifacts = pytest.mark.skipif(
    not (os.path.exists("data/processed/risk_model.pkl") and os.path.exists("data/processed/features.csv")),
    reason="requires locally trained model/features (run `python -m src.train` first)",
)


@requires_local_artifacts
def test_dashboard_loads_without_exceptions():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.metric) == 2
    assert len(at.selectbox) == 1
    assert len(at.dataframe) == 1


@requires_local_artifacts
def test_selecting_a_different_customer_updates_the_score():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=60)

    selectbox = at.selectbox[0]
    first_customer_score = at.metric[1].value

    other_customer = next(o for o in selectbox.options if o != selectbox.value)
    selectbox.select(other_customer).run(timeout=60)

    assert not at.exception
    assert at.metric[1].value != first_customer_score


def test_dashboard_shows_error_when_artifacts_missing(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/risk_model.pkl")
    monkeypatch.setenv("FEATURES_PATH", "/nonexistent/features.csv")

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.error) == 1
    assert len(at.metric) == 0
