"""Generación de predicciones con el modelo entrenado."""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.train import load_model, prepare_xy
from src.utils.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_LABELS = {0: "No Podium", 1: "P1", 2: "P2", 3: "P3"}


def predict(df: pd.DataFrame, model_path: str = "xgb_model.json") -> pd.DataFrame:
    """Genera predicciones de probabilidad y clase para un DataFrame de entrada."""
    model = load_model(model_path)
    X, _ = prepare_xy(df)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    prob_df = pd.DataFrame(probs, columns=[f"Prob_{CLASS_LABELS[i]}" for i in range(4)])
    prob_df["PredictedClass"] = preds
    prob_df["PredictedLabel"] = prob_df["PredictedClass"].map(CLASS_LABELS)

    result = pd.concat([df.reset_index(drop=True), prob_df], axis=1)
    return result


def predict_for_race(df: pd.DataFrame, year: int, round_num: int, model_path: str = "xgb_model.json") -> pd.DataFrame:
    """Filtra una carrera específica y devuelve predicciones ordenadas por probabilidad de podium."""
    race = df[(df["Year"] == year) & (df["RoundNumber"] == round_num)].copy()
    if race.empty:
        logger.warning(f"No se encontraron datos para {year} R{round_num}")
        return pd.DataFrame()
    predicted = predict(race, model_path=model_path)
    # Ordenar por probabilidad de estar en el podio (P1+P2+P3)
    podium_prob = predicted[["Prob_P1", "Prob_P2", "Prob_P3"]].sum(axis=1)
    predicted["PodiumProbability"] = podium_prob
    return predicted.sort_values("PodiumProbability", ascending=False)


if __name__ == "__main__":
    # Ejemplo de uso CLI
    from src.features.build_features import build_all_features

    df = build_all_features()
    preds = predict_for_race(df, 2023, 1)
    print(preds[["Abbreviation", "TeamName", "PredictedLabel", "PodiumProbability"]].head(10))
