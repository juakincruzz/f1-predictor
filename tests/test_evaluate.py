"""Tests de las métricas por carrera, la referencia de la parrilla y el
walk-forward fuera de muestra."""

import numpy as np
import pandas as pd

from src.models import evaluate
from src.models.evaluate import race_metrics, summarize_by_year


def _race(**overrides) -> pd.DataFrame:
    """Carrera inventada de 5 pilotos con resultado conocido.

    Resultado real: A gana, C segundo, B tercero.
    Modelo: probabilidad de podio A > B > C (acierta los 3 pilotos, no el orden)
    y la mayor Prob_P1 es la de A (acierta el ganador).
    Parrilla: B (pole), A, D, C, E -> top 3 B, A, D (2 de 3, no el ganador).
    """
    df = pd.DataFrame(
        {
            "Year": [2024] * 5,
            "RoundNumber": [1] * 5,
            "EventName": ["GP"] * 5,
            "DriverId": ["a", "b", "c", "d", "e"],
            "DriverNumber": ["1", "2", "3", "4", "5"],
            "target": [1, 3, 2, 0, 0],
            "grid_position": [2.0, 1.0, 4.0, 3.0, 5.0],
            "Prob_P1": [0.50, 0.20, 0.10, 0.05, 0.01],
            "Prob_P2": [0.20, 0.25, 0.20, 0.05, 0.01],
            "Prob_P3": [0.10, 0.25, 0.20, 0.10, 0.02],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    df["PodiumProbability"] = df[["Prob_P1", "Prob_P2", "Prob_P3"]].sum(axis=1)
    return df


def test_race_metrics_model():
    m = race_metrics(_race())
    assert m["Model_PodiumHits"] == 3
    assert m["Model_Winner"] is True
    assert m["Model_FullPodium"] is True  # los 3 pilotos, aunque no en orden
    assert m["Model_Perfect"] is False  # predice A, B, C; fue A, C, B


def test_race_metrics_grid_baseline():
    m = race_metrics(_race())
    assert m["Grid_PodiumHits"] == 2  # B y A sí, D no
    assert m["Grid_Winner"] is False  # la pole (B) no ganó
    assert m["Grid_Perfect"] is False


def test_perfect_podium_needs_exact_order():
    # el modelo ordena A, C, B por probabilidad de podio = resultado real
    race = _race(Prob_P3=[0.10, 0.05, 0.30, 0.10, 0.02])
    m = race_metrics(race)
    assert m["Model_PodiumHits"] == 3
    assert m["Model_Perfect"] is True


def test_pit_lane_start_is_back_of_the_grid():
    """En Jolpica grid 0 = salida desde el pit lane, no la pole."""
    m = race_metrics(_race(grid_position=[1.0, 2.0, 3.0, 4.0, 0.0]))
    # si E (grid 0) contara como pole, el top 3 sería E, A, B -> 2 aciertos
    assert m["Grid_PodiumHits"] == 3  # A, B, C
    assert m["Grid_Winner"] is True  # pole = A, que ganó


def test_summarize_by_year_averages_races_and_adds_no_podium_baseline():
    second = _race(RoundNumber=[2] * 5, target=[0, 0, 0, 1, 0])
    second.loc[second["DriverId"] == "e", "target"] = 2
    second.loc[second["DriverId"] == "c", "target"] = 3
    preds = pd.concat([_race(), second], ignore_index=True)
    preds["PredictedClass"] = 0
    summary = summarize_by_year(preds).iloc[0]

    assert summary["Races"] == 2
    # carrera 1: 3 aciertos; carrera 2: podio D, E, C y el modelo dijo A, B, C -> 1
    assert summary["Model_PodiumHits"] == 2.0
    assert summary["Model_WinnerRate"] == 0.5
    # referencia «siempre no podio»: la fracción de filas sin podio
    assert summary["NoPodium_Accuracy"] == (preds["target"] == 0).mean()
    assert summary["Accuracy"] == summary["NoPodium_Accuracy"]  # predijo todo 0


def _synthetic_seasons(years=(2020, 2021, 2022), races=4, drivers=10) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for year in years:
        for rnd in range(1, races + 1):
            grid = rng.permutation(drivers) + 1
            for d in range(drivers):
                rows.append(
                    {
                        "Year": year,
                        "RoundNumber": rnd,
                        "EventName": f"GP {rnd}",
                        "DriverId": f"d{d}",
                        "DriverNumber": str(d),
                        "Abbreviation": f"D{d:02d}",
                        "grid_position": float(grid[d]),
                        "recent_form_avg": rng.uniform(1, 10),
                        "target": int(grid[d]) if grid[d] <= 3 else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_out_of_sample_predictions_never_train_on_the_predicted_season(monkeypatch):
    """Cada temporada se predice con un modelo entrenado solo con años anteriores."""
    seen = []
    real_train = evaluate.train_model

    def spy(train_df, **kwargs):
        seen.append(set(train_df["Year"]))
        return real_train(train_df, **kwargs)

    monkeypatch.setattr(evaluate, "train_model", spy)
    df = _synthetic_seasons()
    oos = evaluate.out_of_sample_predictions(df)

    assert sorted(oos["Year"].unique()) == [2021, 2022]  # 2020 no tiene pasado
    assert seen == [{2020}, {2020, 2021}]
    assert (oos["TrainedUpTo"] < oos["Year"]).all()
    assert {"Prob_P1", "PodiumProbability"} <= set(oos.columns)
