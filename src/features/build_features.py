"""Feature engineering para el predictor de podios de F1."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import config
from src.utils.paths import get_raw_path, get_processed_path
from src.data.collect import load_raw_sessions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_MAP = {1: 1, 2: 2, 3: 3}
NO_PODIUM = 0


def build_targets(race_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna la clase multiclase: 0=no podium, 1=P1, 2=P2, 3=P3."""
    race_df = race_df.copy()
    # La columna de posición final puede variar según FastF1 version
    pos_col = "Position" if "Position" in race_df.columns else "ClassifiedPosition"
    race_df["FinalPosition"] = pd.to_numeric(race_df[pos_col], errors="coerce").astype("Int64")
    race_df["Target"] = race_df["FinalPosition"].map(TARGET_MAP).fillna(NO_PODIUM).astype(int)
    return race_df


def compute_recent_form(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales en las últimas n carreras por piloto."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["RecentFormAvg"] = (
        df.groupby("DriverNumber")["FinalPosition"]
        .shift(1)
        .rolling(window=n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def compute_team_pace(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales de la escudería en las últimas n carreras."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["TeamPaceAvg"] = (
        df.groupby("TeamName")["FinalPosition"]
        .shift(1)
        .rolling(window=n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def compute_track_history(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales del piloto en ese circuito en las últimas n apariciones."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["TrackHistoryAvg"] = (
        df.groupby(["DriverNumber", "EventName"])["FinalPosition"]
        .shift(1)
        .rolling(window=n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def compute_championship_position(df: pd.DataFrame) -> pd.DataFrame:
    """Posición acumulada en el campeonato antes de la carrera."""
    df = df.sort_values(["Year", "RoundNumber"])
    # Puntos por carrera
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df["PointsRace"] = df["FinalPosition"].map(points_map).fillna(0)
    df["CumPoints"] = df.groupby(["Year", "DriverNumber"])["PointsRace"].shift(1).cumsum().fillna(0)
    # Ranking dentro de la temporada hasta la carrera anterior
    df["ChampionshipPosition"] = (
        df.groupby(["Year", "RoundNumber"])["CumPoints"]
        .rank(method="min", ascending=False)
        .fillna(20)
        .astype(int)
    )
    return df


def compute_races_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Número acumulado de carreras disputadas por piloto."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["RacesExperience"] = df.groupby("DriverNumber").cumcount()
    return df


def compute_teammate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Diferencia de posición respecto al compañero de equipo en la carrera anterior."""
    df = df.sort_values(["Year", "RoundNumber"])
    shifted = df.groupby(["Year", "TeamName", "RoundNumber"])["FinalPosition"].shift(1)
    # Simplificación: diferencia absoluta con la media del equipo en la carrera previa
    team_avg = (
        df.groupby(["Year", "TeamName", "RoundNumber"])["FinalPosition"]
        .shift(1)
        .transform("mean")
    )
    df["TeammateGap"] = df["FinalPosition"] - team_avg
    return df


def extract_grid_position(quali_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la posición de salida desde los resultados de clasificación."""
    quali = quali_df.copy()
    grid_col = "Position" if "Position" in quali.columns else "GridPosition"
    quali["GridPosition"] = pd.to_numeric(quali[grid_col], errors="coerce").astype("Int64")
    return quali[["Year", "RoundNumber", "DriverNumber", "GridPosition"]]


def merge_race_and_quali(race_df: pd.DataFrame, quali_df: pd.DataFrame) -> pd.DataFrame:
    """Une resultados de carrera con posición de salida."""
    grid = extract_grid_position(quali_df)
    merged = race_df.merge(grid, on=["Year", "RoundNumber", "DriverNumber"], how="left")
    return merged


def build_features_for_year(year: int) -> pd.DataFrame:
    """Construye el dataset de features para una temporada."""
    race = load_raw_sessions(year, "R")
    quali = load_raw_sessions(year, "Q")

    if race.empty:
        logger.warning(f"No hay datos de carrera para {year}")
        return pd.DataFrame()

    race = build_targets(race)
    race = merge_race_and_quali(race, quali)

    n_form = config.get("features.recent_races_form", 5)
    n_team = config.get("features.team_pace_races", 5)
    n_track = config.get("features.track_history_races", 5)

    race = compute_recent_form(race, n_form)
    race = compute_team_pace(race, n_team)
    race = compute_track_history(race, n_track)
    race = compute_championship_position(race)
    race = compute_races_experience(race)
    race = compute_teammate_gap(race)

    # Selección de columnas de interés
    feature_cols = config.get("features.columns", [])
    base_cols = ["Year", "RoundNumber", "EventName", "DriverNumber", "Abbreviation", "TeamName", "Target"]
    keep_cols = base_cols + feature_cols
    available_cols = [c for c in keep_cols if c in race.columns]
    return race[available_cols].copy()


def build_all_features(start_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    """Concatena features de todas las temporadas en un único DataFrame."""
    start = start_year or config.get("data.start_year")
    end = end_year or config.get("data.end_year")

    frames = []
    for year in range(start, end + 1):
        df_year = build_features_for_year(year)
        if not df_year.empty:
            frames.append(df_year)

    if not frames:
        logger.error("No se generó ningún feature.")
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    # Persistir
    output_path = get_processed_path("features.parquet")
    full.to_parquet(output_path, index=False)
    logger.info(f"Features guardadas en {output_path} ({len(full)} filas)")
    return full


if __name__ == "__main__":
    build_all_features()
