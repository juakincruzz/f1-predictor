"""Entrenamiento del modelo XGBoost multiclase."""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import config
from src.utils.paths import get_processed_path, get_models_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_features() -> pd.DataFrame:
    """Carga el dataset de features procesado."""
    path = get_processed_path("features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Features no encontradas en {path}. Ejecuta 'make features' primero.")
    return pd.read_parquet(path)


def prepare_xy(df: pd.DataFrame):
    """Separa X e y, eliminando columnas no numéricas e IDs."""
    drop_cols = ["target", "Year", "RoundNumber", "EventName", "DriverNumber", "Abbreviation", "TeamName"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"].astype(int)
    # Rellenar NaN con medianas simples
    X = X.fillna(X.median())
    return X, y


def train_model(df: pd.DataFrame | None = None, test_size: float | None = None) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Entrena un XGBClassifier multiclase."""
    if df is None:
        df = load_features()

    X, y = prepare_xy(df)
    test_sz = test_size if test_size is not None else config.get("model.test_size", 0.2)

    if test_sz > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=config.get("model.params.random_state", 42), stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X.iloc[:1], y.iloc[:1]  # dummy para evitar errores

    params = config.get("model.params", {})
    num_class = config.get("model.num_class", 4)
    objective = config.get("model.objective", "multi:softprob")
    eval_metric = config.get("model.eval_metric", "mlogloss")

    model = XGBClassifier(
        objective=objective,
        num_class=num_class,
        eval_metric=eval_metric,
        **params,
    )

    early_stopping = config.get("model.early_stopping_rounds", 20)
    fit_kwargs = {"verbose": False}
    if test_sz > 0:
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        fit_kwargs["early_stopping_rounds"] = early_stopping

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # XGBoost >= 3.x: early_stopping_rounds se pasa en el constructor
        if "early_stopping_rounds" in fit_kwargs:
            model.set_params(early_stopping_rounds=fit_kwargs.pop("early_stopping_rounds"))
        model.fit(X_train, y_train, **fit_kwargs)

    logger.info(f"Modelo entrenado. Mejor iteración: {model.best_iteration}")
    return model, X_train, X_test, y_train, y_test


def save_model(model: XGBClassifier, filename: str = "xgb_model.json") -> Path:
    """Persiste el modelo entrenado."""
    path = get_models_path(filename)
    model.save_model(str(path))
    logger.info(f"Modelo guardado en {path}")
    return path


def load_model(filename: str = "xgb_model.json") -> XGBClassifier:
    """Carga un modelo previamente entrenado."""
    path = get_models_path(filename)
    model = XGBClassifier()
    model.load_model(str(path))
    return model


if __name__ == "__main__":
    df = load_features()
    model, *_ = train_model(df)
    save_model(model)
