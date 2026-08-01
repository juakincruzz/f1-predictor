"""Entrenamiento del modelo XGBoost multiclase con split temporal."""

import logging
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import config
from src.utils.paths import get_processed_path, get_models_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DROP_COLS = [
    "target",
    "Year",
    "RoundNumber",
    "EventName",
    "DriverNumber",
    "Abbreviation",
    "TeamName",
]


def load_features() -> pd.DataFrame:
    """Carga el dataset de features procesado."""
    path = get_processed_path("features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Features no encontradas en {path}. Ejecuta 'make features' primero."
        )
    return pd.read_parquet(path)


def prepare_xy(df: pd.DataFrame, medians: pd.Series | None = None):
    """Separa X e y, eliminando columnas no numéricas e IDs.

    Los NaN se rellenan con `medians` (las del train para evitar leakage);
    si no se pasan, se usan las del propio df como fallback.
    """
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"].astype(int)
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X, y


def train_model(
    df: pd.DataFrame | None = None, test_size: float | None = None
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Entrena un XGBClassifier multiclase.

    Con test_size > 0 usa split TEMPORAL: entrena con años < validation.val_year
    y usa ese año como conjunto de early stopping. Con test_size = 0 (o si no
    hay datos del año de validación) entrena con todo el df sin early stopping.
    """
    if df is None:
        df = load_features()

    test_sz = test_size if test_size is not None else config.get("model.test_size", 0.2)
    val_year = config.get("validation.val_year")

    use_holdout = (
        test_sz > 0
        and val_year is not None
        and "Year" in df.columns
        and (df["Year"] == val_year).any()
        and (df["Year"] < val_year).any()
    )

    if use_holdout:
        train_df = df[df["Year"] < val_year]
        val_df = df[df["Year"] == val_year]
        logger.info(
            f"Split temporal: train < {val_year} ({len(train_df)} filas), val = {val_year} ({len(val_df)} filas)"
        )
    else:
        if test_sz > 0:
            logger.warning(
                f"Sin datos del año de validación {val_year}: entrenando sin early stopping."
            )
        train_df, val_df = df, None

    X_train, y_train = prepare_xy(train_df)
    train_medians = train_df.drop(
        columns=[c for c in DROP_COLS if c in train_df.columns]
    ).median(numeric_only=True)

    if val_df is not None:
        X_test, y_test = prepare_xy(val_df, medians=train_medians)
    else:
        X_test, y_test = (
            X_train.iloc[:1],
            y_train.iloc[:1],
        )  # dummy para mantener la firma

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
    if val_df is not None:
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        fit_kwargs["early_stopping_rounds"] = early_stopping

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # XGBoost >= 3.x: early_stopping_rounds se pasa en el constructor
        if "early_stopping_rounds" in fit_kwargs:
            model.set_params(
                early_stopping_rounds=fit_kwargs.pop("early_stopping_rounds")
            )
        model.fit(X_train, y_train, **fit_kwargs)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        logger.info(f"Modelo entrenado. Mejor iteración: {best_iteration}")
    else:
        logger.info("Modelo entrenado (sin early stopping).")
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
