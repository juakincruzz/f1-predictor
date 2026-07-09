"""Recolección incremental de datos de F1 usando FastF1.

Incluye rate limiting automático y reintentos con backoff para respetar
los límites de la API de FastF1 (500 calls/h).
"""

import logging
import sys
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import config
from src.utils.paths import get_raw_path, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(str(ensure_dir(Path(config.get("data.cache_path")))))  # type: ignore[arg-type]

# Rate limiting: ~500 calls/hour -> 1 call cada 7.2s
# Dejamos margen de seguridad con 10s entre sesiones
DEFAULT_SESSION_DELAY = 10  # segundos entre cada sesión descargada
DEFAULT_RETRY_BASE = 60  # segundos base para reintentos
DEFAULT_MAX_RETRIES = 3


def session_exists(year: int, round_num: int, session_type: str) -> bool:
    """Comprueba si ya existe el Parquet para una sesión."""
    filepath = get_raw_path(f"{year}") / f"R{round_num:02d}_{session_type}.parquet"
    return filepath.exists()


def _download_with_retry(
    year: int,
    round_num: int,
    session_type: str,
    event_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: int = DEFAULT_RETRY_BASE,
) -> pd.DataFrame | None:
    """Descarga una sesión con reintentos ante rate limits."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load()
            df = session.results

            if df is None or df.empty:
                logger.warning(f"Sin datos para {year} R{round_num} {session_type}")
                return None

            df = df.copy()
            df["Year"] = year
            df["RoundNumber"] = round_num
            df["SessionType"] = session_type
            df["EventName"] = event_name
            return df

        except RateLimitExceededError as e:
            last_exception = e
            wait = base_delay * (2 ** attempt)  # 60, 120, 240...
            logger.warning(
                f"Rate limit excedido en {year} R{round_num} {session_type}. "
                f"Esperando {wait}s antes de reintento {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

        except Exception as e:
            last_exception = e
            logger.error(f"Error descargando {year} R{round_num} {session_type}: {e}")
            return None

    logger.error(
        f"Falló tras {max_retries} reintentos: {year} R{round_num} {session_type}. "
        f"Error: {last_exception}"
    )
    return None


def fetch_session(
    year: int,
    round_num: int,
    session_type: str,
    event_name: str = "",
    delay: int = DEFAULT_SESSION_DELAY,
) -> pd.DataFrame | None:
    """Descarga una sesión de FastF1 aplicando rate limiting y reintentos."""
    result = _download_with_retry(year, round_num, session_type, event_name)
    if result is not None and delay > 0:
        time.sleep(delay)
    return result


def save_raw(df: pd.DataFrame, year: int, round_num: int, session_type: str) -> None:
    """Persiste el DataFrame en data/raw/{year}/R{round_num:02d}_{session_type}.parquet"""
    year_dir = get_raw_path(str(year))
    filepath = year_dir / f"R{round_num:02d}_{session_type}.parquet"
    df.to_parquet(filepath, index=False)
    logger.info(f"Guardado: {filepath}")


def collect_year(
    year: int,
    force: bool = False,
    session_delay: int = DEFAULT_SESSION_DELAY,
) -> dict:
    """Descarga todas las carreras y clasificaciones de una temporada (salta testing).
    
    Returns:
        dict con 'processed', 'skipped', 'failed', 'total'.
    """
    logger.info(f"=" * 60)
    logger.info(f"Procesando temporada {year}...")
    logger.info(f"=" * 60)
    
    schedule = fastf1.get_event_schedule(year)
    # Filtrar eventos de testing y similares
    valid_events = schedule[~schedule["EventName"].str.contains("Testing|Test", case=False, na=False)]
    total_events = len(valid_events)

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total": total_events * 2}

    for idx, (_, event) in enumerate(valid_events.iterrows(), 1):
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        
        logger.info(f"[{idx}/{total_events}] {event_name} — R{round_num:02d}")

        # Race (R)
        if not session_exists(year, round_num, "R") or force:
            df_race = fetch_session(year, round_num, "R", event_name, delay=session_delay)
            if df_race is not None:
                save_raw(df_race, year, round_num, "R")
                stats["processed"] += 1
            else:
                stats["failed"] += 1
        else:
            logger.info(f"  {year} R{round_num:02d} R ya existe, saltando.")
            stats["skipped"] += 1

        # Qualifying (Q)
        if not session_exists(year, round_num, "Q") or force:
            df_quali = fetch_session(year, round_num, "Q", event_name, delay=session_delay)
            if df_quali is not None:
                save_raw(df_quali, year, round_num, "Q")
                stats["processed"] += 1
            else:
                stats["failed"] += 1
        else:
            logger.info(f"  {year} R{round_num:02d} Q ya existe, saltando.")
            stats["skipped"] += 1

    logger.info(f"=" * 60)
    logger.info(f"Temporada {year} completada.")
    logger.info(f"  Procesados: {stats['processed']} | Saltados: {stats['skipped']} | Fallados: {stats['failed']}")
    logger.info(f"=" * 60)
    return stats


def collect_range(
    start_year: int | None = None,
    end_year: int | None = None,
    force: bool = False,
    session_delay: int = DEFAULT_SESSION_DELAY,
) -> None:
    """Descarga incremental de un rango de temporadas con rate limiting."""
    start = start_year or config.get("data.start_year")
    end = end_year or config.get("data.end_year")

    grand_total = {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

    for year in range(start, end + 1):
        stats = collect_year(year, force=force, session_delay=session_delay)
        for key in grand_total:
            grand_total[key] += stats.get(key, 0)

    logger.info(f"=" * 60)
    logger.info(f"DESCARGA COMPLETA {start}-{end}")
    logger.info(f"  Procesados: {grand_total['processed']}")
    logger.info(f"  Saltados:   {grand_total['skipped']}")
    logger.info(f"  Fallados:   {grand_total['failed']}")
    logger.info(f"  Total:      {grand_total['total']}")
    logger.info(f"=" * 60)


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
