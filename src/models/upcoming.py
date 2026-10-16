"""Predicción de la próxima carrera del calendario.

La carrera futura se añade al histórico como una fila más por piloto, sin
resultado, y pasa por el mismo build_features_from_frames que el
entrenamiento: como las features históricas usan shift(1), solo ven carreras
anteriores.
"""

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.collect import load_schedule
from src.features.build_features import (
    build_features_from_frames,
    driver_key,
    load_raw_range,
)
from src.models.predict import predict_for_race
from src.utils.config import config

GRID_REAL = "real"
GRID_ESTIMATED = "estimada"
ENTRY_COLUMNS = [
    "DriverNumber",
    "DriverId",
    "Abbreviation",
    "FullName",
    "TeamName",
    "TeamId",
    "TeamColor",
]
# Una ronda sin resultados cuya fecha pasó hace más de esto se da por cancelada
STALE_DAYS = 3


@dataclass
class UpcomingRace:
    year: int
    round_num: int
    event_name: str
    event_date: date
    total_rounds: int


@dataclass
class UpcomingPrediction:
    race: UpcomingRace
    grid_source: str
    predictions: pd.DataFrame


def next_race(
    schedule: pd.DataFrame, completed_rounds: set[int], today: date
) -> UpcomingRace | None:
    """Primera ronda del calendario sin resultado y no caducada."""
    if schedule.empty:
        return None
    sched = schedule.sort_values("RoundNumber")
    dates = pd.to_datetime(sched["EventDate"]).dt.date
    pending = sched[
        ~sched["RoundNumber"].isin(completed_rounds)
        & (dates >= today - timedelta(days=STALE_DAYS))
    ]
    if pending.empty:
        return None
    row = pending.iloc[0]
    return UpcomingRace(
        year=int(row["Year"]),
        round_num=int(row["RoundNumber"]),
        event_name=str(row["EventName"]),
        event_date=pd.Timestamp(row["EventDate"]).date(),
        total_rounds=len(sched),
    )


def _estimated_grid(
    entries: pd.DataFrame, quali_hist: pd.DataFrame, n_recent: int
) -> pd.Series:
    """Parrilla estimada: ranking de la media de las últimas n qualys de cada
    piloto; quien no tiene historial sale al final."""
    if quali_hist.empty:
        avg = pd.Series(np.nan, index=entries.index)
    else:
        q = quali_hist.sort_values(["Year", "RoundNumber"])
        positions = pd.to_numeric(q["Position"], errors="coerce")
        recent = positions.groupby(driver_key(q)).agg(lambda s: s.tail(n_recent).mean())
        avg = driver_key(entries).map(recent)
    return avg.rank(method="first", na_option="bottom")


def build_upcoming_entries(
    race_hist: pd.DataFrame,
    quali_hist: pd.DataFrame,
    race: UpcomingRace,
    n_recent: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Filas de la carrera futura (una por piloto) con su posición de salida.

    Si la clasificación de esa ronda ya existe se usa como parrilla real; si no,
    los pilotos son los de la última carrera y la parrilla se estima.
    """
    quali_now = pd.DataFrame()
    if not quali_hist.empty:
        quali_now = quali_hist[
            (quali_hist["Year"] == race.year)
            & (quali_hist["RoundNumber"] == race.round_num)
        ]

    if not quali_now.empty:
        cols = [c for c in ENTRY_COLUMNS if c in quali_now.columns]
        entries = quali_now[cols].copy()
        entries["GridPosition"] = pd.to_numeric(quali_now["Position"], errors="coerce")
        source = GRID_REAL
    else:
        hist = race_hist.sort_values(["Year", "RoundNumber"])
        last_year, last_round = hist[["Year", "RoundNumber"]].iloc[-1]
        last = hist[(hist["Year"] == last_year) & (hist["RoundNumber"] == last_round)]
        cols = [c for c in ENTRY_COLUMNS if c in last.columns]
        entries = last[cols].copy()
        entries["GridPosition"] = _estimated_grid(entries, quali_hist, n_recent)
        source = GRID_ESTIMATED

    entries = entries.reset_index(drop=True)
    entries["Position"] = np.nan
    entries["Year"] = race.year
    entries["RoundNumber"] = race.round_num
    entries["EventName"] = race.event_name
    entries["SessionType"] = "R"
    return entries, source


def predict_next_race(
    today: date | None = None, model_path: str = "xgb_model.json"
) -> UpcomingPrediction | None:
    """Predice la próxima carrera de la temporada en curso (None si no hay)."""
    today = today or date.today()
    year = today.year
    start = config.get("data.start_year")
    race_hist = load_raw_range(start, year, "R")
    if race_hist.empty:
        return None
    quali_hist = load_raw_range(start, year, "Q")

    completed = set(race_hist.loc[race_hist["Year"] == year, "RoundNumber"])
    race = next_race(load_schedule(year), completed, today)
    if race is None:
        return None

    entries, source = build_upcoming_entries(race_hist, quali_hist, race)
    features = build_features_from_frames(
        pd.concat([race_hist, entries], ignore_index=True), quali_hist
    )
    preds = predict_for_race(features, race.year, race.round_num, model_path=model_path)
    return UpcomingPrediction(race=race, grid_source=source, predictions=preds)


if __name__ == "__main__":
    result = predict_next_race()
    if result is None:
        print("No hay próxima carrera pendiente con datos.")
    else:
        r = result.race
        print(
            f"{r.year} R{r.round_num} {r.event_name} ({r.event_date}) "
            f"— parrilla {result.grid_source}"
        )
        cols = ["Abbreviation", "TeamName", "grid_position", "PodiumProbability"]
        print(result.predictions[cols].head(10).to_string(index=False))
