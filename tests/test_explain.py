"""Tests de explicabilidad del modelo."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.explain import feature_importance, podium_shap


def _model_and_x():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "grid_position": rng.uniform(1, 20, 200),
            "race_round": rng.uniform(0, 1, 200),
        }
    )
    grid = X["grid_position"]
    y = np.select([grid < 2, grid < 3, grid < 4], [1, 2, 3], 0)
    model = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=20)
    model.fit(X, y)
    return model, X


def test_feature_importance_normalized_and_labeled():
    model, _ = _model_and_x()
    imp = feature_importance(model)
    assert set(imp["feature"]) == {"grid_position", "race_round"}
    assert abs(imp["importance"].sum() - 1) < 1e-9
    assert imp.iloc[0]["feature"] == "grid_position"
    assert imp.iloc[0]["label"] == "Posición de salida"


def test_podium_shap_ranks_grid_first():
    model, X = _model_and_x()
    shap_df = podium_shap(model, X.head(20))
    assert list(shap_df.columns) == ["feature", "label", "shap"]
    assert shap_df.iloc[0]["feature"] == "grid_position"
    assert (shap_df["shap"] >= 0).all()
