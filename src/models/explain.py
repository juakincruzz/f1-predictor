"""Explicabilidad del modelo: importancia global y SHAP por carrera.

Usa los valores SHAP nativos de XGBoost (pred_contribs, TreeSHAP exacto), que
no dependen de la compatibilidad de la librería shap con cada versión.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

PODIUM_CLASSES = [1, 2, 3]
FEATURE_LABELS = {
    "grid_position": "Posición de salida",
    "recent_form_avg": "Forma reciente",
    "team_pace_avg": "Ritmo del equipo",
    "track_history_avg": "Historial en el circuito",
    "championship_position": "Posición en el campeonato",
    "races_experience": "Experiencia (carreras)",
    "teammate_gap": "Gap con el compañero",
    "race_round": "Momento de la temporada",
}


def _labeled(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return (
        df[["feature", "label", value_col]]
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def feature_importance(model: XGBClassifier) -> pd.DataFrame:
    """Importancia por ganancia (gain) normalizada a 1."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    names = booster.feature_names or list(scores)
    df = pd.DataFrame(
        {"feature": names, "importance": [scores.get(n, 0.0) for n in names]}
    )
    total = df["importance"].sum()
    if total > 0:
        df["importance"] = df["importance"] / total
    return _labeled(df, "importance")


def podium_shap(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Media de |SHAP| por feature, sumada sobre las clases de podio."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    # (filas, clases, features + sesgo) -> descarta el sesgo
    podium = np.abs(contribs[:, PODIUM_CLASSES, :-1]).sum(axis=1)
    df = pd.DataFrame({"feature": list(X.columns), "shap": podium.mean(axis=0)})
    return _labeled(df, "shap")
