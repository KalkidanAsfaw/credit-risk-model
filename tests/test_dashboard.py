import os

import matplotlib.pyplot as plt
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


@requires_local_artifacts
def test_repeated_reruns_do_not_leak_matplotlib_figures():
    """Regression test: an unclosed portfolio-chart figure previously bled through
    into the SHAP waterfall plot after a customer switch triggered a rerun.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=60)

    selectbox = at.selectbox[0]
    for i in range(5):
        other = selectbox.options[(i * 37) % len(selectbox.options)]
        selectbox.select(other).run(timeout=60)
        assert plt.get_fignums() == [], "matplotlib figure leaked across a dashboard rerun"

    assert not at.exception


def test_dashboard_shows_error_when_artifacts_missing(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/risk_model.pkl")
    monkeypatch.setenv("FEATURES_PATH", "/nonexistent/features.csv")

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.error) == 1
    assert len(at.metric) == 0
