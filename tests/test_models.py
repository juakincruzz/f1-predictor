"""Tests para entrenamiento del modelo."""

import numpy as np
import pandas as pd

from src.models.train import prepare_xy, train_model


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


def _synthetic_years(years, n_per_year=60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = n_per_year * len(years)
    grid = rng.integers(1, 21, n).astype(float)
    return pd.DataFrame(
        {
            "Year": np.repeat(years, n_per_year),
            "RoundNumber": np.tile(np.arange(n_per_year) // 20 + 1, len(years)),
            "DriverNumber": [str(i % 20) for i in range(n)],
            "FullName": ["Driver"] * n,
            "target": np.where(grid <= 3, grid, 0).astype(int),
            "grid_position": grid,
            "recent_form_avg": rng.uniform(1, 20, n),
        }
    )


def test_prepare_xy_ignores_non_feature_columns():
    X, y = prepare_xy(_synthetic_years([2024]))
    assert list(X.columns) == ["grid_position", "recent_form_avg"]
    assert len(y) == 60


def test_refit_full_retrains_on_all_years_with_best_iteration():
    df = _synthetic_years([2023, 2024, 2025])  # val_year = 2025 en config
    es_model, X_train, *_ = train_model(df, test_size=0.2)
    assert len(X_train) == 120  # solo años < 2025
    full_model, *_ = train_model(df, test_size=0.2, refit_full=True)
    assert full_model.get_booster().num_boosted_rounds() == es_model.best_iteration + 1
