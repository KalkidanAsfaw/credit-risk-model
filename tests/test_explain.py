import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from lightgbm import LGBMClassifier  # noqa: E402

from src.explain import compute_shap_values, plot_global_importance, plot_local_explanation  # noqa: E402


def _tiny_model_and_data():
    rng = np.random.default_rng(0)
    n, p = 60, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = np.array([0, 1] * (n // 2))
    model = LGBMClassifier(n_estimators=10, max_depth=2, verbose=-1).fit(X, y)
    return model, X


def test_compute_shap_values_shape():
    model, X = _tiny_model_and_data()
    shap_values = compute_shap_values(model, X)
    assert shap_values.values.shape == (len(X), X.shape[1])


def test_plot_global_importance_creates_file(tmp_path):
    model, X = _tiny_model_and_data()
    shap_values = compute_shap_values(model, X)
    output_path = tmp_path / "global.png"
    plot_global_importance(shap_values, str(output_path))
    assert output_path.exists()


def test_plot_local_explanation_creates_file(tmp_path):
    model, X = _tiny_model_and_data()
    shap_values = compute_shap_values(model, X)
    output_path = tmp_path / "local.png"
    plot_local_explanation(shap_values, 0, str(output_path))
    assert output_path.exists()
