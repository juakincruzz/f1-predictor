"""Tests para entrenamiento del modelo."""

import numpy as np
import pandas as pd

from src.models.train import train_model


def _synthetic_features(n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Year": np.repeat([2020, 2021], n_rows // 2),
            "RoundNumber": np.tile(range(1, n_rows // 2 + 1), 2)[:n_rows],
            "EventName": ["GP"] * n_rows,
            "DriverNumber": [str(i % 20) for i in range(n_rows)],
            "Abbreviation": ["DRV"] * n_rows,
            "TeamName": ["Team"] * n_rows,
            "target": rng.integers(0, 4, n_rows),
            "grid_position": rng.integers(1, 21, n_rows).astype(float),
            "recent_form_avg": rng.uniform(1, 20, n_rows),
            "race_round": rng.uniform(0, 1, n_rows),
        }
    )


def test_train_model_without_holdout_does_not_crash():
    """train_model con test_size=0.0 (como usa evaluate.py) no debe lanzar
    AttributeError por best_iteration cuando no hay early stopping."""
    df = _synthetic_features()
    model, *_ = train_model(df, test_size=0.0)
    assert model is not None
    # El modelo debe poder predecir
    preds = model.predict(df[["grid_position", "recent_form_avg", "race_round"]])
    assert len(preds) == len(df)
