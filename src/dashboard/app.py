"""Dashboard Streamlit del F1 Podium Predictor (estética rojo Fórmula / plata)."""

import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dashboard import components as ui
from src.dashboard.theme import RED, SILVER, apply_theme, style_chart
from src.data.jolpica import collect_year_jolpica
from src.features.build_features import build_features_for_year
from src.models.explain import feature_importance, podium_shap
from src.models.predict import predict_for_race
from src.models.train import load_model, prepare_xy
from src.models.upcoming import GRID_REAL, predict_next_race
from src.utils.config import config
from src.utils.paths import get_models_path, get_raw_path, get_reports_path

MODEL_FILE = "xgb_model.json"

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")
apply_theme()
ui.header("F1 Podium", "Predictor", "XGBoost multiclase · validación walk-forward")

if toast := st.session_state.pop("toast", None):
    st.toast(toast)

model_path = get_models_path(MODEL_FILE)
model_mtime = model_path.stat().st_mtime if model_path.exists() else None


@st.cache_data(show_spinner="Calculando features...")
def season_features(year: int) -> pd.DataFrame:
    return build_features_for_year(year)


@st.cache_data(show_spinner="Calculando predicción...")
def race_predictions(year: int, round_num: int, mtime: float) -> pd.DataFrame:
    return predict_for_race(
        season_features(year), year, round_num, model_path=MODEL_FILE
    )


@st.cache_data(show_spinner="Prediciendo la próxima carrera...")
def upcoming(today: date, mtime: float):
    return predict_next_race(today=today, model_path=MODEL_FILE)


@st.cache_resource
def trained_model(mtime: float):
    return load_model(MODEL_FILE)


def available_years() -> list[int]:
    start = config.get("data.start_year", 2010)
    end = config.get("data.end_year", 2026)
    return [
        y
        for y in range(start, end + 1)
        if any(get_raw_path(str(y)).glob("*_R.parquet"))
    ]


def bar_chart(df: pd.DataFrame, value: str, color: str, title: str) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusEnd=2, height=18)
        .encode(
            x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=4)),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip(f"{value}:Q", format=".3f"),
            ],
        )
        .properties(height=34 * len(df) + 30)
    )
    return style_chart(chart)


# --- Barra lateral -----------------------------------------------------------
with st.sidebar:
    ui.section("Selección de carrera")
    years = available_years()
    year = st.selectbox("Temporada", years[::-1]) if years else None
    feats = season_features(year) if year else pd.DataFrame()

    round_labels: dict[int, str] = {}
    if not feats.empty:
        rounds = feats[["RoundNumber", "EventName"]].drop_duplicates("RoundNumber")
        round_labels = {
            int(r.RoundNumber): f"R{int(r.RoundNumber):02d} · {r.EventName}"
            for r in rounds.sort_values("RoundNumber").itertuples()
        }
    round_num = (
        st.selectbox(
            "Gran Premio",
            list(round_labels),
            index=len(round_labels) - 1,
            format_func=round_labels.get,
        )
        if round_labels
        else None
    )

    ui.section("Datos")
    if st.button("🔄 Actualizar temporada actual", width="stretch"):
        current = date.today().year
        with st.spinner(f"Descargando {current} desde Jolpica..."):
            try:
                stats = collect_year_jolpica(current)
            # red caída, HTTP de error o respuesta con otro formato
            except (
                requests.RequestException,
                ConnectionError,
                KeyError,
                ValueError,
            ) as e:
                st.error(f"No se pudo actualizar: {e}")
            else:
                st.cache_data.clear()
                st.session_state["toast"] = (
                    f"Temporada {current} actualizada: "
                    f"{stats['processed']} sesiones nuevas"
                )
                st.rerun()
    st.caption("Datos: Jolpica / FastF1 · Modelo: XGBoost")
    st.markdown("[Repositorio en GitHub](https://github.com/juakincruzz/f1-predictor)")

if not years:
    st.warning("No hay datos descargados. Ejecuta `make data` y `make features`.")

tab_next, tab_pred, tab_hist, tab_feat = st.tabs(
    ["🏁 Próxima carrera", "🔮 Predicción", "📊 Histórico", "🧠 Features"]
)

# --- Próxima carrera ---------------------------------------------------------
with tab_next:
    if model_mtime is None:
        ui.model_missing()
    elif (up := upcoming(date.today(), model_mtime)) is None:
        st.info(
            "No hay próxima carrera pendiente: la temporada ha terminado o falta el "
            "calendario. Pulsa «Actualizar temporada actual» o ejecuta `make update`."
        )
    else:
        race = up.race
        real_grid = up.grid_source == GRID_REAL
        ui.stat_cards(
            [
                ("Gran Premio", race.event_name),
                ("Fecha", race.event_date.strftime("%d/%m/%Y")),
                ("Ronda", f"{race.round_num} / {race.total_rounds}"),
                ("Parrilla", "Real (qualy)" if real_grid else "Estimada"),
            ]
        )
        if not real_grid:
            st.caption(
                "Aún no hay clasificación: la parrilla se estima con la media de las "
                "últimas 5 qualys de cada piloto. Actualiza los datos tras la qualy "
                "para afinar la predicción."
            )
        ui.section("Podio previsto")
        ui.podium_cards(up.predictions)
        ui.section("Probabilidades por piloto")
        ui.prediction_table(up.predictions)

# --- Predicción de una carrera disputada -------------------------------------
with tab_pred:
    if model_mtime is None:
        ui.model_missing()
    elif round_num is None:
        st.info("Selecciona una temporada con datos en la barra lateral.")
    else:
        preds = race_predictions(year, round_num, model_mtime)
        actual = preds[preds["target"] > 0].sort_values("target")
        hits = len(set(preds.head(3)["Abbreviation"]) & set(actual["Abbreviation"]))
        winner = actual.iloc[0]["Abbreviation"] if not actual.empty else "—"
        ui.stat_cards(
            [
                ("Carrera", f"{year} · {round_labels[round_num].split(' · ', 1)[1]}"),
                ("Aciertos de podio", f"{hits} / 3"),
                ("Ganador real", winner),
                ("Favorito del modelo", preds.iloc[0]["Abbreviation"]),
            ]
        )
        ui.section("Podio previsto")
        ui.podium_cards(preds)
        ui.section("Podio real")
        ui.podium_cards(actual, prob_label="El modelo le daba")
        ui.section("Detalle por piloto")
        ui.prediction_table(preds, with_actual=True)
        st.caption(
            "El modelo final se entrena con todas las temporadas, así que las "
            "carreras pasadas son datos que ya ha visto. Las métricas fuera de "
            "muestra están en la pestaña Histórico."
        )

# --- Histórico walk-forward --------------------------------------------------
with tab_hist:
    results_path = get_reports_path("walk_forward_results.csv")
    if not results_path.exists():
        st.info("Ejecuta `make evaluate` para generar las métricas walk-forward.")
    else:
        hist = pd.read_csv(results_path)
        ui.stat_cards(
            [
                ("Temporadas evaluadas", ", ".join(map(str, hist["TestYear"]))),
                ("Accuracy media", f"{hist['Accuracy'].mean():.1%}"),
                ("F1 macro media", f"{hist['F1Macro'].mean():.3f}"),
                ("Log loss media", f"{hist['LogLoss'].mean():.3f}"),
            ]
        )
        ui.section("Métricas por temporada (entrenando solo con años anteriores)")
        long = hist.melt(
            id_vars="TestYear",
            value_vars=["Accuracy", "F1Macro"],
            var_name="Métrica",
            value_name="Valor",
        )
        line = (
            alt.Chart(long)
            .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=3)
            .encode(
                x=alt.X(
                    "TestYear:O",
                    title="Temporada de test",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("Valor:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=["Accuracy", "F1Macro"], range=[SILVER, RED]
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["TestYear", "Métrica", alt.Tooltip("Valor:Q", format=".3f")],
            )
            .properties(height=320)
        )
        st.altair_chart(style_chart(line), theme=None, width="stretch")
        st.dataframe(hist, hide_index=True, width="stretch")

# --- Features ----------------------------------------------------------------
with tab_feat:
    if model_mtime is None:
        ui.model_missing()
    else:
        model = trained_model(model_mtime)
        col_imp, col_shap = st.columns(2)
        with col_imp:
            ui.section("Importancia global (gain)")
            imp = feature_importance(model)
            st.altair_chart(
                bar_chart(imp, "importance", RED, "Peso en el modelo"),
                theme=None,
                width="stretch",
            )
        with col_shap:
            if round_num is None:
                st.info("Selecciona una carrera para ver su explicación SHAP.")
            else:
                ui.section(f"SHAP · {round_labels[round_num]}")
                race_rows = feats[feats["RoundNumber"] == round_num]
                X, _ = prepare_xy(race_rows)
                st.altair_chart(
                    bar_chart(
                        podium_shap(model, X), "shap", SILVER, "Impacto medio en podio"
                    ),
                    theme=None,
                    width="stretch",
                )
        st.caption(
            "La importancia global mide cuánto aporta cada variable al modelo. SHAP "
            "mide cuánto movió cada variable la probabilidad de podio en la carrera "
            "seleccionada (media del valor absoluto sobre P1, P2 y P3)."
        )
