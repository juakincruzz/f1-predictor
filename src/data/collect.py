"""Recolección incremental de datos de F1 usando FastF1."""

import logging
import sys
from pathlib import Path

import fastf1
import pandas as pd

# Añadir src al path para imports absolutos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import config
from src.utils.paths import get_raw_path, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(str(ensure_dir(Path(config.get("data.cache_path")))))  # type: ignore[arg-type]


def session_exists(year: int, round_num: int, session_type: str) -> bool:
    """Comprueba si ya existe el Parquet para una sesión."""
    filepath = get_raw_path(f"{year}") / f"R{round_num:02d}_{session_type}.parquet"
    return filepath.exists()


def fetch_session(year: int, round_num: int, session_type: str, event_name: str = "") -> pd.DataFrame | None:
    """Descarga una sesión de FastF1 y la devuelve como DataFrame.

    session_type puede ser: 'R' (Race), 'Q' (Qualifying), 'FP1', 'FP2', 'FP3', 'S' (Sprint)
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load()

        df = session.results

        if df is None or df.empty:
            logger.warning(f"Sin datos para {year} R{round_num} {session_type}")
            return None

        # Añadir metadatos
        df = df.copy()
        df["Year"] = year
        df["RoundNumber"] = round_num
        df["SessionType"] = session_type
        df["EventName"] = event_name
        return df

    except Exception as e:
        logger.error(f"Error descargando {year} R{round_num} {session_type}: {e}")
        return None


def save_raw(df: pd.DataFrame, year: int, round_num: int, session_type: str) -> None:
    """Persiste el DataFrame en data/raw/{year}/R{round_num:02d}_{session_type}.parquet"""
    year_dir = get_raw_path(str(year))
    filepath = year_dir / f"R{round_num:02d}_{session_type}.parquet"
    df.to_parquet(filepath, index=False)
    logger.info(f"Guardado: {filepath}")


def collect_year(year: int, force: bool = False) -> None:
    """Descarga todas las carreras y clasificaciones de una temporada (salta testing)."""
    logger.info(f"Procesando temporada {year}...")
    schedule = fastf1.get_event_schedule(year)
    # Filtrar eventos de testing y similares
    valid_events = schedule[~schedule["EventName"].str.contains("Testing|Test", case=False, na=False)]
    total_events = len(valid_events)

    for _, event in valid_events.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        # Race (R)
        if not session_exists(year, round_num, "R") or force:
            df_race = fetch_session(year, round_num, "R", event_name)
            if df_race is not None:
                save_raw(df_race, year, round_num, "R")
        else:
            logger.info(f"{year} R{round_num:02d} R ya existe, saltando.")

        # Qualifying (Q)
        if not session_exists(year, round_num, "Q") or force:
            df_quali = fetch_session(year, round_num, "Q", event_name)
            if df_quali is not None:
                save_raw(df_quali, year, round_num, "Q")
        else:
            logger.info(f"{year} R{round_num:02d} Q ya existe, saltando.")

    logger.info(f"Temporada {year} completada. {total_events} eventos procesados.")


def collect_range(start_year: int | None = None, end_year: int | None = None, force: bool = False) -> None:
    """Descarga incremental de un rango de temporadas."""
    start = start_year or config.get("data.start_year")
    end = end_year or config.get("data.end_year")

    for year in range(start, end + 1):
        collect_year(year, force=force)


def load_raw_sessions(year: int, session_type: str = "R") -> pd.DataFrame:
    """Carga todos los Parquet de un año y tipo de sesión en un solo DataFrame."""
    year_dir = get_raw_path(str(year))
    files = sorted(year_dir.glob(f"*_{session_type}.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    collect_range()
