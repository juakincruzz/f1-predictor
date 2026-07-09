"""Dashboard Streamlit para el F1 Podium Predictor."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.build_features import build_features_for_year
from src.models.predict import predict_for_race
from src.utils.config import config
from src.utils.paths import get_models_path

st.set_page_config(page_title="F1 Podium Predictor", layout="wide")

st.title("🏎️ F1 Podium Predictor")
st.markdown("Predicción de podios con XGBoost multiclase.")

# Sidebar: selección de carrera
st.sidebar.header("Selección de Carrera")
years = list(range(config.get("data.start_year", 2010), config.get("data.end_year", 2026) + 1))
default_year = config.get("dashboard.default_year", 2024)
selected_year = st.sidebar.selectbox("Temporada", years, index=years.index(default_year) if default_year in years else len(years) - 1)

# Intentar inferir rondas disponibles para el año
rounds = list(range(1, 25))
selected_round = st.sidebar.selectbox("Ronda", rounds, index=0)

mode = st.sidebar.radio("Modo", ["Pre-carrera", "Datos en vivo"])

# Verificar si existe un modelo entrenado
model_path = get_models_path("xgb_model.json")
model_exists = model_path.exists()
if not model_exists:
    st.sidebar.warning("⚠️ No hay modelo entrenado. Ejecuta `make train` primero.")

# Pestañas
tab_pred, tab_hist, tab_feat, tab_live = st.tabs(
    ["🔮 Predicción", "📊 Histórico", "🧠 Features", "📡 En Vivo"]
)

with tab_pred:
    st.subheader(f"Predicción: {selected_year} - Ronda {selected_round}")
    if not model_exists:
        st.error("No se encontró modelo entrenado. Ejecuta `make train` para generar `models/xgb_model.json`.")
    else:
        df_year = build_features_for_year(selected_year)
        if df_year.empty:
            st.warning("No hay datos procesados para esta temporada. Ejecuta 'make data' y 'make features'.")
        else:
            preds = predict_for_race(df_year, selected_year, selected_round, model_path=str(model_path))
            if preds.empty:
                st.warning("No se encontraron datos de esta carrera.")
            else:
                cols_display = [
                    "Abbreviation", "TeamName", "grid_position", "PredictedLabel",
                    "Prob_P1", "Prob_P2", "Prob_P3", "PodiumProbability"
                ]
                cols_available = [c for c in cols_display if c in preds.columns]
                st.dataframe(preds[cols_available].head(20), use_container_width=True)

                # Top 3 más probable
                st.markdown("### 🏆 Top 3 Probabilidades de Podio")
                top3 = preds.head(3)
                for i, row in top3.iterrows():
                    st.write(f"**{row['Abbreviation']}** ({row['TeamName']}) — {row['PredictedLabel']} — Podio: {row['PodiumProbability']:.1%}")

with tab_hist:
    st.subheader("Validación Histórica")
    st.markdown("Comparación de predicciones vs resultados reales por temporada.")
    # Cargar resultados de evaluación si existen
    reports_path = Path(__file__).resolve().parent.parent.parent / "reports" / "walk_forward_results.csv"
    if reports_path.exists():
        hist_df = pd.read_csv(reports_path)
        st.line_chart(hist_df.set_index("TestYear")[["Accuracy", "F1Macro"]])
    else:
        st.info("Ejecuta 'make evaluate' para generar métricas históricas.")

with tab_feat:
    st.subheader("Importancia de Features")
    st.markdown("Análisis de importancia del modelo XGBoost.")
    # Placeholder para SHAP o feature importance
    st.info("Próximamente: integración de SHAP y gráficos de importancia.")

with tab_live:
    st.subheader("Datos en Vivo")
    st.markdown("Monitoreo de sesiones en curso (requiere FastF1 live donde esté disponible).")
    st.info("Funcionalidad avanzada para futuras versiones.")

st.sidebar.markdown("---")
st.sidebar.markdown("[Repositorio GitHub](https://github.com/juakincruzz/f1-predictor)")
