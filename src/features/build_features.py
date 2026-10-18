"""Feature engineering para el predictor de podios de F1.

Las features históricas (forma reciente, ritmo de equipo, historial en circuito,
experiencia) se calculan sobre TODO el histórico multi-temporada, siempre con
shift(1) dentro del grupo para no filtrar el resultado de la propia carrera.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.collect import load_raw_sessions, load_schedule
from src.utils.config import config
from src.utils.paths import get_processed_path, get_raw_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_MAP = {1: 1, 2: 2, 3: 3}
NO_PODIUM = 0


def build_targets(race_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna la clase multiclase: 0=no podium, 1=P1, 2=P2, 3=P3."""
    race_df = race_df.copy()
    pos_col = "Position" if "Position" in race_df.columns else "ClassifiedPosition"
    race_df["final_position"] = pd.to_numeric(race_df[pos_col], errors="coerce").astype(
        "Int64"
    )
    race_df["target"] = (
        race_df["final_position"].map(TARGET_MAP).fillna(NO_PODIUM).astype(int)
    )
    return race_df


def driver_key(df: pd.DataFrame) -> pd.Series:
    """Identificador estable del piloto: DriverId (el dorsal puede cambiar entre
    temporadas y repetirse entre épocas), con fallback a DriverNumber."""
    if "DriverId" in df.columns:
        ids = df["DriverId"].replace("", pd.NA)
        return ids.fillna(df["DriverNumber"])
    return df["DriverNumber"]


def track_key(df: pd.DataFrame) -> pd.Series:
    """Identificador estable del circuito: CircuitId (el nombre del GP cambia
    con patrocinadores y un mismo nombre puede saltar de circuito), con
    fallback a EventName."""
    if "CircuitId" in df.columns:
        ids = df["CircuitId"].replace("", pd.NA)
        return ids.fillna(df["EventName"])
    return df["EventName"]


def _rolling_prev_mean(grouped, n: int) -> pd.Series:
    """Media móvil de las n observaciones ANTERIORES, calculada por grupo."""
    return grouped.transform(
        lambda s: s.shift(1).rolling(window=n, min_periods=1).mean()
    )


def compute_recent_form(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales en las últimas n carreras por piloto."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["recent_form_avg"] = _rolling_prev_mean(
        df.groupby(driver_key(df))["final_position"], n
    )
    return df


def compute_team_pace(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales de la escudería en las últimas n carreras."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["team_pace_avg"] = _rolling_prev_mean(
        df.groupby("TeamName")["final_position"], n
    )
    return df


def compute_track_history(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Media de posiciones finales del piloto en ese circuito en las últimas n apariciones."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["track_history_avg"] = _rolling_prev_mean(
        df.groupby([driver_key(df), track_key(df)])["final_position"], n
    )
    return df


def compute_championship_position(df: pd.DataFrame) -> pd.DataFrame:
    """Posición en el campeonato antes de la carrera.

    Usa los puntos reales (columna Points: incluye vuelta rápida y medios
    puntos) y, si faltan, la tabla 25-18-15... sobre la posición final.
    """
    df = df.sort_values(["Year", "RoundNumber"], kind="stable")
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    table = df["final_position"].map(points_map).astype(float).fillna(0)
    if "Points" in df.columns:
        df["points_race"] = pd.to_numeric(df["Points"], errors="coerce").fillna(table)
    else:
        df["points_race"] = table

    season_driver = [df["Year"], driver_key(df)]
    df["cum_points"] = (
        df.groupby(season_driver)["points_race"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )
    df["championship_position"] = (
        df.groupby(["Year", "RoundNumber"])["cum_points"]
        .rank(method="min", ascending=False)
        .fillna(20)
        .astype(int)
    )
    return df


def compute_races_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Número acumulado de carreras disputadas por piloto en todo el histórico."""
    df = df.sort_values(["Year", "RoundNumber"])
    df["races_experience"] = df.groupby(driver_key(df)).cumcount()
    return df


def compute_teammate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Mi resultado en mi carrera anterior menos el de mis compañeros de equipo
    en la suya (media si hay más de uno). Solo usa carreras pasadas: ni el
    resultado de la carrera actual ni los futuros entran en la feature."""
    df = df.sort_values(["Year", "RoundNumber"], kind="stable")
    prev = df.groupby(driver_key(df))["final_position"].shift(1).astype(float)
    team = [df["Year"], df["RoundNumber"], df["TeamName"]]
    team_sum = prev.groupby(team).transform("sum")
    team_count = prev.groupby(team).transform("count")
    mates = team_count - prev.notna()
    mates_avg = (team_sum - prev.fillna(0)) / mates.where(mates > 0)
    df["teammate_gap"] = prev - mates_avg
    return df


def extract_grid_position(quali_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la posición de clasificación como fallback de posición de salida."""
    if quali_df.empty:
        return pd.DataFrame(
            columns=["Year", "RoundNumber", "DriverNumber", "grid_position_quali"]
        )
    quali = quali_df.copy()
    grid_col = "Position" if "Position" in quali.columns else "GridPosition"
    quali["grid_position_quali"] = pd.to_numeric(quali[grid_col], errors="coerce")
    return quali[["Year", "RoundNumber", "DriverNumber", "grid_position_quali"]]


def merge_race_and_quali(race_df: pd.DataFrame, quali_df: pd.DataFrame) -> pd.DataFrame:
    """Posición de salida: GridPosition del resultado de carrera (incluye
    penalizaciones) como fuente primaria; posición de quali como fallback."""
    race_df = race_df.copy()
    if "GridPosition" in race_df.columns:
        race_df["grid_position"] = pd.to_numeric(
            race_df["GridPosition"], errors="coerce"
        )
    else:
        race_df["grid_position"] = float("nan")

    grid = extract_grid_position(quali_df)
    if not grid.empty:
        race_df = race_df.merge(
            grid, on=["Year", "RoundNumber", "DriverNumber"], how="left"
        )
        race_df["grid_position"] = race_df["grid_position"].fillna(
            race_df["grid_position_quali"]
        )
        race_df = race_df.drop(columns=["grid_position_quali"])

    race_df["grid_position"] = pd.to_numeric(race_df["grid_position"], errors="coerce")
    return race_df


def add_race_round(df: pd.DataFrame) -> pd.DataFrame:
    """Número de carrera en la temporada (normalizado 0-1).

    Si se conoce la longitud del calendario (columna TotalRounds, que escribe
    el colector Jolpica) se usa esa; si no, la última ronda con datos. Así una
    temporada en curso no parece terminar en la última carrera disputada.
    """
    total = df.groupby("Year")["RoundNumber"].transform("max")
    if "TotalRounds" in df.columns:
        known = pd.to_numeric(df["TotalRounds"], errors="coerce")
        total = known.groupby(df["Year"]).transform("max").fillna(total)
    df["race_round"] = df["RoundNumber"] / total
    return df


def build_features_from_frames(
    race_df: pd.DataFrame, quali_df: pd.DataFrame
) -> pd.DataFrame:
    """Construye el dataset de features a partir de frames crudos multi-temporada."""
    if race_df.empty:
        return pd.DataFrame()

    df = build_targets(race_df)
    df = merge_race_and_quali(df, quali_df)
    df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

    n_form = config.get("features.recent_races_form", 5)
    n_team = config.get("features.team_pace_races", 5)
    n_track = config.get("features.track_history_races", 5)

    df = compute_recent_form(df, n_form)
    df = compute_team_pace(df, n_team)
    df = compute_track_history(df, n_track)
    df = compute_championship_position(df)
    df = compute_races_experience(df)
    df = compute_teammate_gap(df)
    df = add_race_round(df)

    feature_cols = config.get("features.columns", [])
    base_cols = [
        "Year",
        "RoundNumber",
        "EventName",
        "CircuitId",
        "DriverNumber",
        "DriverId",
        "Abbreviation",
        "FullName",
        "TeamName",
        "TeamColor",
        "target",
    ]
    keep_cols = base_cols + feature_cols
    available_cols = [c for c in keep_cols if c in df.columns]
    return df[available_cols].copy()


def attach_circuit_ids(df: pd.DataFrame, years: range) -> pd.DataFrame:
    """Añade CircuitId desde el calendario de cada temporada (si no lo trae)."""
    if df.empty or "CircuitId" in df.columns:
        return df
    schedules = [load_schedule(y) for y in years]
    schedules = [
        s[["Year", "RoundNumber", "CircuitId"]]
        for s in schedules
        if not s.empty and "CircuitId" in s.columns
    ]
    if not schedules:
        return df
    circuits = pd.concat(schedules, ignore_index=True)
    return df.merge(circuits, on=["Year", "RoundNumber"], how="left")


def load_raw_range(start: int, end: int, session_type: str = "R") -> pd.DataFrame:
    """Carga y concatena los Parquet crudos de un rango de temporadas."""
    years = range(start, end + 1)
    frames = [load_raw_sessions(year, session_type) for year in years]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return attach_circuit_ids(pd.concat(frames, ignore_index=True), years)


def _fresh_processed_features(year: int) -> pd.DataFrame | None:
    """features.parquet si es más nuevo que todos los crudos hasta `year` y
    contiene ese año; None si hay que recalcular."""
    path = get_processed_path("features.parquet")
    if not path.exists():
        return None
    built = path.stat().st_mtime
    for y in range(config.get("data.start_year"), year + 1):
        if any(
            f.stat().st_mtime > built for f in get_raw_path(str(y)).glob("*.parquet")
        ):
            return None
    df = pd.read_parquet(path)
    return df if (df["Year"] == year).any() else None


def build_features_for_year(year: int) -> pd.DataFrame:
    """Features de una temporada, con contexto histórico de temporadas previas.

    Reutiliza data/processed/features.parquet si está al día con los crudos;
    si no, recalcula desde los crudos (sin escribir el parquet).
    """
    cached = _fresh_processed_features(year)
    if cached is not None:
        return cached[cached["Year"] == year].reset_index(drop=True)

    start = config.get("data.start_year")
    race = load_raw_range(start, year, "R")
    quali = load_raw_range(start, year, "Q")

    if race.empty or race[race["Year"] == year].empty:
        logger.warning(f"No hay datos de carrera para {year}")
        return pd.DataFrame()

    full = build_features_from_frames(race, quali)
    return full[full["Year"] == year].reset_index(drop=True)


def build_all_features(
    start_year: int | None = None, end_year: int | None = None
) -> pd.DataFrame:
    """Construye las features de todas las temporadas sobre el histórico completo."""
    start = start_year or config.get("data.start_year")
    end = end_year or config.get("data.end_year")

    race = load_raw_range(start, end, "R")
    quali = load_raw_range(start, end, "Q")

    if race.empty:
        logger.error("No se generó ningún feature: no hay datos crudos de carrera.")
        return pd.DataFrame()

    full = build_features_from_frames(race, quali)
    output_path = get_processed_path("features.parquet")
    full.to_parquet(output_path, index=False)
    logger.info(f"Features guardadas en {output_path} ({len(full)} filas)")
    return full


if __name__ == "__main__":
    build_all_features()
