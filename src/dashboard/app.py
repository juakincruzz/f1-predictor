"""Dashboard Streamlit del F1 Podium Predictor (estética «muro de boxes»).

Tres secciones (Carrera, Temporada, Modelo) elegidas arriba; solo se ejecuta la
abierta. La sección, la temporada y la ronda viven en session_state y se
reflejan en la URL (?seccion=&anio=&ronda=) para recargar sin perderse.
"""

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dashboard import components as ui
from src.dashboard.circuit_map import circuit_code, circuit_svg
from src.dashboard.components import BAD, BEST, GOOD, OFF, WARN, Kpi
from src.dashboard.race_hero import render_hero
from src.dashboard.replay_component import render_replay
from src.dashboard.theme import (
    GREEN,
    PURPLE,
    RED,
    SILVER,
    TEXT,
    YELLOW,
    apply_theme,
    style_chart,
)
from src.data import openf1
from src.data.collect import load_schedule
from src.data.jolpica import collect_year_jolpica, fetch_races, parse_schedule
from src.data.replay import build_replay
from src.features.build_features import build_features_for_year
from src.models.explain import feature_importance, podium_shap
from src.models.predict import predict
from src.models.train import load_model, prepare_xy
from src.models.upcoming import GRID_REAL, next_race, predict_next_race
from src.utils.config import config
from src.utils.paths import get_models_path, get_raw_path, get_reports_path

MODEL_FILE = "xgb_model.json"
SECTIONS = ["Carrera", "Temporada", "Modelo"]
SECTION_ICONS = {
    "Carrera": ":material/sports_score:",
    "Temporada": ":material/calendar_month:",
    "Modelo": ":material/memory:",
}
DONE, NEXT, PENDING, NO_RESULTS = "done", "next", "pending", "no_results"
HIT_TONES = {3: BEST, 2: GOOD, 1: WARN, 0: BAD}
HIT_COLORS = {3: PURPLE, 2: GREEN, 1: YELLOW, 0: RED}
STATUS_TAGS = {
    DONE: ("Disputada", "var(--text)"),
    NEXT: ("Próxima carrera", BEST),
    PENDING: ("Pendiente", OFF),
    NO_RESULTS: ("Sin resultados", BAD),
}
MONTHS = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
]
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

if toast := st.session_state.pop("toast", None):
    st.toast(toast)

model_path = get_models_path(MODEL_FILE)
model_mtime = model_path.stat().st_mtime if model_path.exists() else None
today = date.today()


@dataclass
class RoundInfo:
    round: int
    name: str
    code: str
    event_date: date | None
    start: str
    place: str
    circuit_id: str
    status: str
    hits: int | None = None


# --- Datos (cacheados) -------------------------------------------------------
@st.cache_data(show_spinner="Calculando features...")
def season_features(year: int) -> pd.DataFrame:
    return build_features_for_year(year)


@st.cache_data(show_spinner="Calculando predicciones de la temporada...")
def season_predictions(year: int, mtime: float) -> pd.DataFrame:
    """Predicciones de todas las rondas disputadas de la temporada."""
    feats = season_features(year)
    if feats.empty:
        return pd.DataFrame()
    preds = predict(feats, model=trained_model(mtime))
    preds["PodiumProbability"] = preds[["Prob_P1", "Prob_P2", "Prob_P3"]].sum(axis=1)
    return preds


@st.cache_data(show_spinner="Prediciendo la próxima carrera...")
def upcoming(day: date, mtime: float):
    return predict_next_race(today=day, model_path=MODEL_FILE)


@st.cache_resource
def trained_model(mtime: float):
    return load_model(MODEL_FILE)


@st.cache_data(show_spinner="Preparando la repetición...")
def replay_payload(year: int, round_num: int) -> dict:
    session, frames = openf1.load_race(year, round_num)
    return build_replay(session, frames)


def race_event_date(year: int, round_num: int) -> date:
    """Fecha del GP según el calendario (lo descarga de Jolpica si falta)."""
    schedule = load_schedule(year)
    if schedule.empty:
        schedule = parse_schedule(fetch_races(year), year)
        schedule.to_parquet(get_raw_path(str(year)) / "schedule.parquet", index=False)
    row = schedule[schedule["RoundNumber"] == round_num]
    if row.empty:
        raise LookupError(f"La ronda {round_num} no está en el calendario de {year}")
    return pd.Timestamp(row["EventDate"].iloc[0]).date()


def available_years() -> list[int]:
    start = config.get("data.start_year", 2010)
    end = config.get("data.end_year", 2026)
    return [
        y
        for y in range(start, end + 1)
        if any(get_raw_path(str(y)).glob("*_R.parquet"))
    ]


def completed_rounds(year: int) -> set[int]:
    return {
        int(p.name[1:3]) for p in get_raw_path(str(year)).glob("R[0-9][0-9]_R.parquet")
    }


def sorted_by_podium(preds: pd.DataFrame) -> pd.DataFrame:
    return preds.sort_values("PodiumProbability", ascending=False)


def driver_col(df: pd.DataFrame) -> str:
    return "DriverId" if "DriverId" in df.columns else "Abbreviation"


def round_hits(preds: pd.DataFrame) -> dict[int, int]:
    """Aciertos de podio por ronda: pilotos del top-3 del modelo que acabaron en el podio."""
    key = driver_col(preds)
    hits = {}
    for rnd, race in preds.groupby("RoundNumber"):
        actual = set(race.loc[race["target"] > 0, key])
        if actual:
            top = set(race.nlargest(3, "PodiumProbability")[key])
            hits[int(rnd)] = len(top & actual)
    return hits


def fmt_date(d: date | None, weekday: bool = False) -> str:
    if d is None:
        return "fecha por confirmar"
    text = f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return f"{WEEKDAYS[d.weekday()]} {text}" if weekday else text


def season_rounds(year: int, next_round: int | None) -> list[RoundInfo]:
    """Rondas de la temporada con su estado, desde el calendario o las features."""
    done = completed_rounds(year)
    schedule = load_schedule(year)
    if schedule.empty:
        feats = season_features(year)
        if feats.empty:
            return []
        schedule = feats[["RoundNumber", "EventName"]].drop_duplicates("RoundNumber")
    rounds = []
    for row in schedule.sort_values("RoundNumber").itertuples():
        rnd = int(row.RoundNumber)
        event_date = (
            pd.Timestamp(row.EventDate).date() if hasattr(row, "EventDate") else None
        )
        if rnd in done:
            status = DONE
        elif rnd == next_round:
            status = NEXT
        elif event_date is not None and event_date < today:
            status = NO_RESULTS
        else:
            status = PENDING
        place = ", ".join(
            str(v)
            for v in (getattr(row, "Locality", ""), getattr(row, "Country", ""))
            if v and not pd.isna(v)
        )
        circuit_id = str(getattr(row, "CircuitId", "") or "")
        rounds.append(
            RoundInfo(
                round=rnd,
                name=str(row.EventName),
                code=circuit_code(
                    circuit_id, getattr(row, "Country", "") or row.EventName
                ),
                event_date=event_date,
                start=str(getattr(row, "EventTime", "") or ""),
                place=place,
                circuit_id=circuit_id,
                status=status,
            )
        )
    return rounds


def round_tone(r: RoundInfo) -> str:
    if r.status == NEXT:
        return "var(--white)"
    if r.status == DONE:
        return (
            HIT_TONES.get(r.hits, "var(--muted)")
            if r.hits is not None
            else "var(--muted)"
        )
    return OFF


def bar_chart(df: pd.DataFrame, value: str, color: str, title: str) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusEnd=2, height=16)
        .encode(
            x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=4)),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip(f"{value}:Q", format=".3f"),
            ],
        )
        .properties(height=30 * len(df) + 30)
    )
    return style_chart(chart)


# --- Estado: sección, temporada y ronda (sincronizados con la URL) ----------
next_info = None
years = available_years()
for y in sorted({*years, today.year}, reverse=True):
    sched = load_schedule(y)
    if not sched.empty and (nr := next_race(sched, completed_rounds(y), today)):
        next_info = nr
        break
year_options = sorted({*years, *([next_info.year] if next_info else [])}, reverse=True)


def next_round_of(year: int) -> int | None:
    return next_info.round_num if next_info and next_info.year == year else None


def default_round(year: int) -> int | None:
    if (nr := next_round_of(year)) is not None:
        return nr
    done = completed_rounds(year)
    return max(done) if done else None


def _param(name: str, valid) -> int | str | None:
    value = st.query_params.get(name)
    if value is None:
        return None
    if value.isdigit():
        value = int(value)
    return value if value in valid else None


ss = st.session_state
if "section" not in ss:
    ss.section = _param("seccion", SECTIONS) or "Carrera"
    ss.year = _param("anio", year_options) or (
        next_info.year if next_info else (year_options[0] if year_options else None)
    )
    ss.round = _param("ronda", range(1, 40)) if ss.year else None
    if ss.round is None and ss.year:
        ss.round = default_round(ss.year)


def _on_nav() -> None:
    if ss._nav is None:
        ss._nav = ss.section
    else:
        ss.section = ss._nav


def _on_year() -> None:
    ss.year = ss._year
    ss.round = default_round(ss.year)


def _on_race() -> None:
    if ss._race is None:
        ss._race = ss.round
    else:
        ss.round = ss._race


# --- Barra superior y navegación --------------------------------------------
leds = [
    (
        ("Modelo OK", "var(--green)", False)
        if model_mtime
        else ("Sin modelo", "var(--red)", True)
    )
]
if years:
    last_year = max(years)
    leds.append(
        (
            f"Datos {last_year} · R{max(completed_rounds(last_year)):02d}",
            "var(--green)",
            False,
        )
    )
else:
    leds.append(("Sin datos", "var(--red)", True))
if next_info:
    leds.append((f"Próxima R{next_info.round_num:02d}", "var(--purple)", True))
ui.topbar(leds)

col_nav, col_btn = st.columns([5, 1.4], vertical_alignment="center")
with col_nav:
    ss._nav = ss.section
    st.segmented_control(
        "Sección",
        SECTIONS,
        format_func=lambda s: f"{SECTION_ICONS[s]} {s}",
        key="_nav",
        on_change=_on_nav,
        label_visibility="collapsed",
    )
with col_btn:
    if st.button(
        "Actualizar datos",
        icon=":material/sync:",
        width="stretch",
        help=f"Descarga de Jolpica las carreras nuevas de {today.year}",
    ):
        with st.spinner(f"Descargando {today.year} desde Jolpica..."):
            try:
                stats = collect_year_jolpica(today.year)
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
                ss.toast = (
                    f"Temporada {today.year} actualizada: "
                    f"{stats['processed']} sesiones nuevas"
                )
                st.rerun()

section = ss.section
st.query_params.update(
    {"seccion": section}
    | ({"anio": str(ss.year)} if ss.year else {})
    | ({"ronda": str(ss.round)} if ss.round else {})
)


def year_picker() -> None:
    ss._year = ss.year
    st.selectbox("Temporada", year_options, key="_year", on_change=_on_year)


def race_strip(rounds: list[RoundInfo]) -> None:
    """Tira de calendario: una pastilla por ronda, coloreada por aciertos."""
    labels = {r.round: f"{r.round:02d} {r.code}" for r in rounds}
    tones = "".join(
        f".st-key-_race button:nth-of-type({i}){{--tone:{round_tone(r)}}}"
        for i, r in enumerate(rounds, 1)
    )
    next_css = "".join(
        f".st-key-_race button:nth-of-type({i}) p::after"
        "{content:' ▸';color:var(--purple)}"
        for i, r in enumerate(rounds, 1)
        if r.status == NEXT
    )
    st.markdown(f"<style>{tones}{next_css}</style>", unsafe_allow_html=True)
    ss._race = ss.round if ss.round in labels else None
    st.pills(
        "Gran Premio",
        list(labels),
        format_func=labels.get,
        key="_race",
        on_change=_on_race,
    )
    ui.legend(
        [
            (BEST, "3/3 en el podio"),
            (GOOD, "2/3"),
            (WARN, "1/3"),
            (BAD, "0/3"),
            ("var(--white)", "Próxima"),
            (OFF, "Pendiente"),
        ]
    )


# --- Carrera -----------------------------------------------------------------
def tab_prediction(r: RoundInfo, preds: pd.DataFrame, up) -> None:
    if r.status == NEXT:
        if model_mtime is None:
            ui.model_missing()
            return
        if up is None or up.race.round_num != r.round:
            ui.empty(
                "No se pudo preparar la predicción",
                "Falta la lista de inscritos de la última carrera.",
                command="make update",
            )
            return
        real_grid = up.grid_source == GRID_REAL
        render_hero(
            {
                "meta": f"Ronda {r.round} / {up.race.total_rounds} · {r.place}",
                "name": r.name,
                "grid_real": real_grid,
                "grid_label": (
                    "Parrilla real (qualy)" if real_grid else "Parrilla estimada"
                ),
                "start": f"{r.event_date.isoformat()}T{r.start}" if r.start else None,
                "date_label": fmt_date(r.event_date, weekday=True),
            }
        )
        race_preds = up.predictions
        if not real_grid:
            st.caption(
                "Aún no hay clasificación: la parrilla se estima con la media de las "
                "últimas 5 qualys de cada piloto. Pulsa «Actualizar datos» tras la qualy "
                "para afinar la predicción."
            )
        podium_and_circuit(r, race_preds, None)
        ui.section(
            "Torre de probabilidades",
            "Todos los pilotos, ordenados por su probabilidad de acabar en el podio.",
        )
        ui.timing_tower(race_preds)
        ui.legend(ui.SECTOR_LEGEND[:3])
        return

    if r.status != DONE:
        ui.empty(
            (
                "Esta carrera aún no se ha disputado"
                if r.status == PENDING
                else "No hay resultados de esta carrera"
            ),
            "La predicción se abre cuando es la próxima del calendario. "
            + (
                f"Ahora mismo es la R{next_info.round_num:02d} ({next_info.event_name})."
                if next_info
                else ""
            ),
            OFF if r.status == PENDING else BAD,
        )
        return
    if model_mtime is None:
        ui.model_missing()
        return

    preds = sorted_by_podium(preds[preds["RoundNumber"] == r.round])
    actual = preds[preds["target"] > 0].sort_values("target")
    key = driver_col(preds)
    hits = len(set(preds.head(3)[key]) & set(actual[key]))
    fav, winner = preds.iloc[0], actual.iloc[0] if not actual.empty else None
    winner_ok = winner is not None and fav[key] == winner[key]
    ui.kpis(
        [
            Kpi(
                "Aciertos de podio",
                hits,
                "pilotos del top-3 previsto en el podio",
                HIT_TONES[hits],
                "/3",
            ),
            Kpi(
                "Favorito del modelo",
                str(fav["Abbreviation"]),
                "acertó al ganador" if winner_ok else "no ganó",
                BEST if winner_ok else BAD,
            ),
            Kpi(
                "Ganador real",
                str(winner["Abbreviation"]) if winner is not None else "—",
                str(winner["TeamName"]) if winner is not None else "",
            ),
            Kpi(
                "Prob. que le daba al ganador",
                (
                    round(float(winner["PodiumProbability"]) * 100)
                    if winner is not None
                    else 0
                ),
                "de acabar en el podio",
                suffix="%",
            ),
        ]
    )
    podium_and_circuit(r, preds, actual)
    ui.section(
        "Torre de probabilidades",
        "La columna Real compara la predicción con el resultado: morado = posición "
        "exacta, verde = en el podio, amarillo = sorpresa que el modelo no vio, rojo = fallo.",
    )
    ui.timing_tower(preds, with_actual=True)
    st.caption(
        "El modelo final se entrena con todas las temporadas, así que las carreras "
        "pasadas son datos que ya ha visto. Las métricas honestas (fuera de muestra) "
        "están en la sección Modelo."
    )


def podium_and_circuit(r: RoundInfo, preds: pd.DataFrame, actual) -> None:
    col_main, col_side = st.columns([3, 2], gap="medium")
    with col_main:
        ui.section("Podio previsto", "Las tres mayores probabilidades de podio.")
        ui.podium_cards(preds)
        if actual is not None:
            ui.section(
                "Podio real", "Y la probabilidad que el modelo le daba a cada uno."
            )
            ui.podium_cards(actual, prob_label="El modelo le daba")
    with col_side:
        ui.section("Circuito")
        if svg := circuit_svg(r.circuit_id):
            ui.circuit_card(svg, r.place, fmt_date(r.event_date))
        else:
            ui.empty("Sin trazado", "No hay dibujo de este circuito.", OFF)
        ui.section("Colores")
        ui.legend(ui.SECTOR_LEGEND)


def tab_replay(r: RoundInfo, year: int) -> None:
    if r.status != DONE:
        ui.empty(
            "Repetición no disponible",
            "La repetición existe cuando la carrera ya se ha disputado.",
            OFF,
        )
    elif year < openf1.FIRST_YEAR:
        ui.empty(
            "Repetición no disponible",
            f"Usa datos de OpenF1, que existen desde {openf1.FIRST_YEAR}. "
            "Elige una temporada más reciente.",
            OFF,
        )
    elif openf1.is_downloaded(year, r.round):
        render_replay(replay_payload(year, r.round))
        st.caption(
            "Datos de OpenF1 (posiciones a 1 Hz interpoladas). Espacio para "
            "reproducir o pausar; arrastra la barra para saltar a cualquier momento."
        )
    else:
        ui.empty(
            "Repetición sin descargar",
            "La primera vez se descargan los datos de la carrera desde OpenF1 (plan "
            "gratuito: 30 peticiones por minuto, 1-2 minutos). Después quedan "
            "guardados y la repetición se abre al instante.",
            WARN,
        )
        if st.button(
            "Cargar repetición", icon=":material/play_arrow:", key="load_replay"
        ):
            bar = st.progress(0.0, text="Buscando la sesión en OpenF1...")

            def report(i: int, total: int, label: str) -> None:
                bar.progress(
                    min(i / max(total, 1), 1.0), text=f"Descargando {label}..."
                )

            try:
                openf1.download_race(
                    year, r.round, race_event_date(year, r.round), report
                )
            except (
                requests.RequestException,
                ConnectionError,
                LookupError,
                KeyError,
                ValueError,
            ) as e:
                st.error(f"No se pudo cargar la repetición: {e}")
            else:
                st.rerun()


def tab_why(r: RoundInfo, year: int, up) -> None:
    if model_mtime is None:
        ui.model_missing()
        return
    if r.status == NEXT and up is not None and up.race.round_num == r.round:
        rows = up.predictions
    elif r.status == DONE:
        feats = season_features(year)
        rows = feats[feats["RoundNumber"] == r.round]
    else:
        ui.empty(
            "Sin explicación",
            "Solo se explican carreras disputadas o la próxima.",
            OFF,
        )
        return
    ui.section(
        "Qué movió la predicción",
        "Impacto medio de cada variable en la probabilidad de podio de esta carrera "
        "(SHAP, media del valor absoluto sobre P1, P2 y P3).",
    )
    X, _ = prepare_xy(rows)
    st.altair_chart(
        bar_chart(
            podium_shap(trained_model(model_mtime), X),
            "shap",
            SILVER,
            "Impacto medio en podio",
        ),
        theme=None,
        width="stretch",
    )


def page_race() -> None:
    if not year_options:
        ui.empty(
            "No hay datos descargados",
            "Descarga las temporadas y calcula las features para empezar.",
            BAD,
            "make data && make features",
        )
        return
    year = ss.year
    preds = season_predictions(year, model_mtime) if model_mtime else pd.DataFrame()
    rounds = season_rounds(year, next_round_of(year))
    if not preds.empty:
        hits = round_hits(preds)
        for r in rounds:
            r.hits = hits.get(r.round)

    col_year, col_strip = st.columns([1, 6], vertical_alignment="top")
    with col_year:
        year_picker()
    with col_strip:
        race_strip(rounds)

    r = next((x for x in rounds if x.round == ss.round), None)
    if r is None:
        ui.empty(
            "Elige un Gran Premio", "Pulsa una ronda de la tira de calendario.", OFF
        )
        return
    tag, tone = STATUS_TAGS[r.status]
    ui.context(
        "Viendo",
        f"{year} · R{r.round:02d} · {r.name}",
        " · ".join(x for x in (r.place, fmt_date(r.event_date)) if x),
        tag,
        tone,
    )
    up = upcoming(today, model_mtime) if model_mtime and r.status == NEXT else None
    t_pred, t_replay, t_why = st.tabs(["Predicción", "Repetición", "Por qué"])
    with t_pred:
        tab_prediction(r, preds, up)
    with t_replay:
        tab_replay(r, year)
    with t_why:
        tab_why(r, year, up)


# --- Temporada ---------------------------------------------------------------
def page_season() -> None:
    if not year_options:
        ui.empty("No hay datos descargados", "", BAD, "make data && make features")
        return
    col_year, _ = st.columns([1, 6])
    with col_year:
        year_picker()
    year = ss.year
    if model_mtime is None:
        ui.model_missing()
        return
    preds = season_predictions(year, model_mtime)
    rounds = season_rounds(year, next_round_of(year))
    hits = round_hits(preds) if not preds.empty else {}
    if not hits:
        ui.empty(
            f"Sin carreras disputadas en {year}",
            "Cuando haya resultados aparecerá aquí el balance del modelo.",
            OFF,
        )
        return
    for r in rounds:
        r.hits = hits.get(r.round)

    total = len(hits)
    key = driver_col(preds)
    winners = sum(
        sorted_by_podium(race).iloc[0][key]
        == race.loc[race["target"] == 1, key].squeeze()
        for _, race in preds.groupby("RoundNumber")
        if (race["target"] == 1).any()
    )
    perfect = sum(1 for h in hits.values() if h == 3)
    ui.section(
        f"Balance {year}",
        "Cuántos pilotos del podio previsto acabaron de verdad en el podio, carrera a carrera.",
    )
    ui.kpis(
        [
            Kpi("Carreras disputadas", total, f"de {len(rounds)} en el calendario"),
            Kpi(
                "Aciertos de podio",
                round(100 * sum(hits.values()) / (3 * total)),
                f"{sum(hits.values())} de {3 * total} pilotos",
                GOOD,
                "%",
            ),
            Kpi("Ganadores acertados", winners, f"de {total} carreras", BEST),
            Kpi("Podios perfectos", perfect, "3 de 3 en el podio", BEST),
        ]
    )
    chart_df = pd.DataFrame(
        [
            {"Ronda": f"{r.round:02d} {r.code}", "GP": r.name, "Aciertos": r.hits}
            for r in rounds
            if r.hits is not None
        ]
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Ronda:N", sort=None, title=None, axis=alt.Axis(labelAngle=-60)),
            y=alt.Y(
                "Aciertos:Q",
                scale=alt.Scale(domain=[0, 3]),
                axis=alt.Axis(tickCount=3, values=[0, 1, 2, 3]),
            ),
            color=alt.Color(
                "Aciertos:O",
                scale=alt.Scale(
                    domain=list(HIT_COLORS), range=list(HIT_COLORS.values())
                ),
                legend=None,
            ),
            tooltip=["GP", "Aciertos"],
        )
        .properties(height=220)
    )
    st.altair_chart(style_chart(chart), theme=None, width="stretch")
    ui.section(
        "Ronda a ronda", "●●○ = dos de los tres pilotos previstos acabaron en el podio."
    )
    ui.season_grid(
        [
            {
                "round": r.round,
                "code": r.code,
                "name": r.name,
                "tone": round_tone(r),
                "dots": r.hits is not None,
                "mark": (
                    "●" * r.hits + "○" * (3 - r.hits)
                    if r.hits is not None
                    else STATUS_TAGS[r.status][0]
                ),
            }
            for r in rounds
        ]
    )
    ui.legend(
        [
            (BEST, "3/3"),
            (GOOD, "2/3"),
            (WARN, "1/3"),
            (BAD, "0/3"),
            ("var(--white)", "Próxima"),
        ]
    )
    st.caption(
        "Se usa el modelo final, que ya ha visto estas carreras: mide cómo se ajusta, "
        "no cómo predice carreras nuevas. Para eso, la sección Modelo."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        done = [r for r in rounds if r.status == DONE]
        pick = st.selectbox(
            "Abrir una carrera",
            [r.round for r in done],
            index=len(done) - 1,
            format_func=lambda n: next(
                f"R{r.round:02d} · {r.name}" for r in done if r.round == n
            ),
            width=320,
        )
        if st.button("Ver carrera", icon=":material/arrow_forward:"):
            ss.section, ss.round = "Carrera", pick
            st.rerun()


# --- Modelo ------------------------------------------------------------------
def page_model() -> None:
    results_path = get_reports_path("walk_forward_results.csv")
    ui.section(
        "Validación walk-forward",
        "Cada temporada se predice con un modelo entrenado solo con las anteriores: "
        "así se mide cómo acierta con carreras que no ha visto.",
    )
    if not results_path.exists():
        ui.empty(
            "Sin métricas", "Aún no se ha evaluado el modelo.", WARN, "make evaluate"
        )
    else:
        hist = pd.read_csv(results_path)
        ui.kpis(
            [
                Kpi("Temporadas evaluadas", ", ".join(map(str, hist["TestYear"]))),
                Kpi(
                    "Accuracy media",
                    round(hist["Accuracy"].mean() * 100),
                    "aciertos de clase",
                    GOOD,
                    "%",
                ),
                Kpi(
                    "F1 macro media",
                    f"{hist['F1Macro'].mean():.3f}",
                    "equilibrio entre clases",
                ),
                Kpi(
                    "Log loss media", f"{hist['LogLoss'].mean():.3f}", "menos es mejor"
                ),
            ]
        )
        long = hist.melt(
            id_vars="TestYear",
            value_vars=["Accuracy", "F1Macro"],
            var_name="Métrica",
            value_name="Valor",
        )
        line = (
            alt.Chart(long)
            .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "TestYear:O", title="Temporada de test", axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y("Valor:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=["Accuracy", "F1Macro"], range=[TEXT, PURPLE]
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["TestYear", "Métrica", alt.Tooltip("Valor:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(style_chart(line), theme=None, width="stretch")
        with st.expander("Ver tabla de métricas"):
            st.dataframe(hist, hide_index=True, width="stretch")

    ui.section(
        "Qué mira el modelo",
        "Importancia global de cada variable (gain normalizado). Para ver qué pesó en "
        "una carrera concreta, abre la pestaña «Por qué» en Carrera.",
    )
    if model_mtime is None:
        ui.model_missing()
        return
    st.altair_chart(
        bar_chart(
            feature_importance(trained_model(model_mtime)),
            "importance",
            RED,
            "Peso en el modelo",
        ),
        theme=None,
        width="stretch",
    )


{"Carrera": page_race, "Temporada": page_season, "Modelo": page_model}[section]()
st.caption(
    "Datos: Jolpica · OpenF1 · Modelo: XGBoost · "
    "[Repositorio en GitHub](https://github.com/juakincruzz/f1-predictor)"
)
