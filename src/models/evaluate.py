"""Evaluación del modelo con validación temporal walk-forward.

Cada temporada se predice con un modelo entrenado solo con las anteriores
(ventana expansiva). Además de las métricas de clasificación, se mide lo que
importa en un podio, carrera a carrera, frente a dos referencias:

- Top 3 de la parrilla: el podio son los tres primeros de la parrilla de
  salida. Es la referencia que hay que batir.
- Siempre «no podio»: solo para poner la accuracy en contexto (~85 % de las
  filas no son podio).
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.build_features import driver_key
from src.models.predict import predict
from src.models.train import train_model
from src.utils.config import config
from src.utils.paths import get_reports_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROB_COLS = ["Prob_No Podium", "Prob_P1", "Prob_P2", "Prob_P3"]
OOS_FILE = "oos_predictions.parquet"
RESULTS_FILE = "walk_forward_results.csv"
RACES_FILE = "walk_forward_races.csv"
SUMMARY_FILE = "walk_forward_summary.json"
PLOT_FILE = "metrics_by_year.png"


def out_of_sample_predictions(
    df: pd.DataFrame, years: list[int] | None = None
) -> pd.DataFrame:
    """Predicciones de cada temporada con un modelo entrenado solo con las
    anteriores (Year < Y). Por defecto, todas las temporadas con pasado.

    Las NaN se rellenan con las medianas de los años de entrenamiento (las
    guarda train_model en el modelo y las usa predict). La columna
    TrainedUpTo indica el último año con el que se entrenó cada predicción.
    """
    if years is None:
        years = sorted(df["Year"].unique())[1:]
    frames = []
    for year in years:
        train_df = df[df["Year"] < year]
        test_df = df[df["Year"] == year]
        if train_df.empty or test_df.empty:
            logger.warning(f"Año {year}: datos insuficientes. Saltando.")
            continue
        model, *_ = train_model(train_df, test_size=0.0)
        preds = predict(test_df, model=model)
        preds["PodiumProbability"] = preds[["Prob_P1", "Prob_P2", "Prob_P3"]].sum(
            axis=1
        )
        preds["TrainedUpTo"] = int(train_df["Year"].max())
        frames.append(preds)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _podium_scores(predicted: list, winner, actual: list) -> dict:
    """Pilotos del podio acertados (0-3), ganador acertado, podio completo
    (los 3 pilotos, en cualquier orden) y podio perfecto (en orden exacto)."""
    hits = len(set(predicted) & set(actual))
    return {
        "PodiumHits": hits,
        "Winner": bool(winner == actual[0]),
        "FullPodium": hits == 3,
        "Perfect": predicted == actual,
    }


def race_metrics(race: pd.DataFrame) -> dict:
    """Métricas de una carrera para el modelo y para la parrilla.

    Modelo: podio previsto = los 3 con más Prob_P1+P2+P3, en ese orden (el
    mismo que enseña el panel); ganador = la mayor Prob_P1.
    Parrilla: podio = los 3 primeros de la parrilla de salida, en orden;
    ganador = la pole. Grid 0 (salida desde el pit lane) cuenta como el final.
    """
    key = driver_key(race)
    actual = list(key[race.loc[race["target"] > 0].sort_values("target").index])

    model_top = list(key[race.nlargest(3, "PodiumProbability").index])
    model_winner = key[race["Prob_P1"].idxmax()]

    grid = race["grid_position"].where(race["grid_position"] > 0, np.inf)
    grid_order = grid.sort_values(kind="stable").index
    grid_top = list(key[grid_order[:3]])

    scores = {
        "Model": _podium_scores(model_top, model_winner, actual),
        "Grid": _podium_scores(grid_top, grid_top[0], actual),
    }
    return {f"{who}_{k}": v for who, s in scores.items() for k, v in s.items()}


def per_race_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Una fila por carrera con resultado (las que tienen podio real)."""
    rows = []
    for (year, rnd), race in preds.groupby(["Year", "RoundNumber"]):
        if (race["target"] > 0).sum() < 3:
            continue  # carrera sin resultado completo
        rows.append(
            {
                "Year": int(year),
                "RoundNumber": int(rnd),
                "EventName": race["EventName"].iloc[0],
                **race_metrics(race),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_year(preds: pd.DataFrame) -> pd.DataFrame:
    """Métricas por temporada: por carrera (medias) y de clasificación."""
    races = per_race_metrics(preds)
    rows = []
    for year, season in preds.groupby("Year"):
        r = races[races["Year"] == year]
        y = season["target"].astype(int)
        probs = season[PROB_COLS].to_numpy() if set(PROB_COLS) <= set(season) else None
        rows.append(
            {
                "TestYear": int(year),
                "Races": len(r),
                "Model_PodiumHits": r["Model_PodiumHits"].mean(),
                "Grid_PodiumHits": r["Grid_PodiumHits"].mean(),
                "Model_WinnerRate": r["Model_Winner"].mean(),
                "Grid_WinnerRate": r["Grid_Winner"].mean(),
                "Model_FullPodiumRate": r["Model_FullPodium"].mean(),
                "Grid_FullPodiumRate": r["Grid_FullPodium"].mean(),
                "Model_PerfectRate": r["Model_Perfect"].mean(),
                "Grid_PerfectRate": r["Grid_Perfect"].mean(),
                "Accuracy": accuracy_score(y, season["PredictedClass"]),
                "NoPodium_Accuracy": (y == 0).mean(),
                "F1Macro": f1_score(
                    y, season["PredictedClass"], average="macro", zero_division=0
                ),
                "LogLoss": (
                    log_loss(y, probs, labels=[0, 1, 2, 3])
                    if probs is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward con ventana expansiva sobre validation.test_years."""
    preds = out_of_sample_predictions(df, config.get("validation.test_years", []))
    if preds.empty:
        return pd.DataFrame()
    results = summarize_by_year(preds)
    for row in results.itertuples():
        logger.info(
            f"Test {row.TestYear} -> pilotos del podio {row.Model_PodiumHits:.2f}/3 "
            f"(parrilla {row.Grid_PodiumHits:.2f}), ganador {row.Model_WinnerRate:.0%} "
            f"(parrilla {row.Grid_WinnerRate:.0%}), accuracy {row.Accuracy:.3f} "
            f"(siempre «no podio» {row.NoPodium_Accuracy:.3f})"
        )
    return results


def overall_summary(races: pd.DataFrame) -> dict:
    """Medias sobre todas las carreras de test (no la media de medias anuales)."""
    out = {"Races": len(races)}
    for who in ("Model", "Grid"):
        out[f"{who}_PodiumHits"] = float(races[f"{who}_PodiumHits"].mean())
        out[f"{who}_WinnerRate"] = float(races[f"{who}_Winner"].mean())
        out[f"{who}_FullPodiumRate"] = float(races[f"{who}_FullPodium"].mean())
        out[f"{who}_PerfectRate"] = float(races[f"{who}_Perfect"].mean())
    return out


def plot_metrics(results_df: pd.DataFrame) -> Path:
    """Pilotos del podio acertados por temporada: modelo frente a parrilla."""
    path = get_reports_path(PLOT_FILE)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    ax.bar(x - 0.2, results_df["Model_PodiumHits"], 0.4, label="Modelo")
    ax.bar(x + 0.2, results_df["Grid_PodiumHits"], 0.4, label="Top 3 de la parrilla")
    ax.set_xticks(x, results_df["TestYear"].astype(str))
    ax.set_ylim(0, 3)
    ax.set_ylabel("Pilotos del podio acertados (de 3)")
    ax.set_title("Walk-forward: modelo frente a parrilla")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Gráfico guardado en {path}")
    return path


def evaluate_full(df: pd.DataFrame | None = None) -> dict:
    """Walk-forward completo: predicciones fuera de muestra de todas las
    temporadas (las lee el panel) y métricas de validation.test_years."""
    if df is None:
        from src.features.build_features import build_all_features

        df = build_all_features()

    oos = out_of_sample_predictions(df)
    oos_path = get_reports_path(OOS_FILE)
    oos.to_parquet(oos_path, index=False)
    logger.info(f"Predicciones fuera de muestra guardadas en {oos_path}")

    test_years = config.get("validation.test_years", [])
    test_preds = oos[oos["Year"].isin(test_years)]
    results = summarize_by_year(test_preds)
    races = per_race_metrics(test_preds)

    csv_path = get_reports_path(RESULTS_FILE)
    results.to_csv(csv_path, index=False)
    races.to_csv(get_reports_path(RACES_FILE), index=False)

    summary = {
        "by_year": results.to_dict(orient="records"),
        "overall": overall_summary(races),
    }
    json_path = get_reports_path(SUMMARY_FILE)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Resumen: {summary['overall']}")

    plot_metrics(results)
    return {"results": results, "paths": {"csv": csv_path, "json": json_path}}


if __name__ == "__main__":
    evaluate_full()
