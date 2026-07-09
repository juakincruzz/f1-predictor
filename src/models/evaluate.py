"""Evaluación del modelo con validación temporal walk-forward."""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.train import prepare_xy, train_model
from src.models.predict import predict
from src.utils.config import config
from src.utils.paths import get_reports_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def walk_forward_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Entrena hasta el año N y evalúa en N+1, iterando."""
    results = []
    train_until = config.get("validation.train_until")
    test_years = config.get("validation.test_years", [])

    for test_year in test_years:
        train_df = df[df["Year"] <= train_until]
        test_df = df[df["Year"] == test_year]

        if train_df.empty or test_df.empty:
            logger.warning(f"Año {test_year}: datos insuficientes. Saltando.")
            continue

        model, _, _, _, _ = train_model(train_df, test_size=0.0)
        X_test, y_test = prepare_xy(test_df)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        ll = log_loss(y_test, probs, labels=[0, 1, 2, 3])

        logger.info(f"Test {test_year} -> Accuracy: {acc:.3f}, F1-macro: {f1:.3f}, LogLoss: {ll:.3f}")
        results.append({
            "TestYear": test_year,
            "Accuracy": acc,
            "F1Macro": f1,
            "LogLoss": ll,
        })

    return pd.DataFrame(results)


def plot_metrics(results_df: pd.DataFrame) -> Path:
    """Genera gráfico de métricas por temporada de test."""
    path = get_reports_path("metrics_by_year.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results_df, x="TestYear", y="Accuracy", marker="o", label="Accuracy", ax=ax)
    sns.lineplot(data=results_df, x="TestYear", y="F1Macro", marker="s", label="F1 Macro", ax=ax)
    ax.set_title("Validación Temporal - Métricas por Temporada")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Gráfico guardado en {path}")
    return path


def evaluate_full(df: pd.DataFrame | None = None) -> dict:
    """Ejecuta la evaluación completa y guarda reportes."""
    if df is None:
        from src.features.build_features import build_all_features
        df = build_all_features()

    wf_results = walk_forward_evaluation(df)

    # Guardar CSV de resultados
    csv_path = get_reports_path("walk_forward_results.csv")
    wf_results.to_csv(csv_path, index=False)

    # Guardar JSON resumen
    summary = wf_results.to_dict(orient="records")
    json_path = get_reports_path("walk_forward_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Gráfico
    plot_metrics(wf_results)

    return {"results": wf_results, "paths": {"csv": csv_path, "json": json_path}}


if __name__ == "__main__":
    evaluate_full()
