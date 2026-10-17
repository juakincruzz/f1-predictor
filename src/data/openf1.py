"""Descarga de datos de carrera desde OpenF1 para la repetición animada.

El plan gratuito de OpenF1 sirve todas las sesiones desde 2023, sin
autenticación y con un límite de 30 peticiones por minuto. Si se define la
variable de entorno OPENF1_TOKEN (plan de pago), se envía como Bearer y el
mismo código sirve para datos en directo.

Cada carrera se guarda en data/openf1/{year}_R{round}/ como parquet; la
primera descarga tarda ~1-2 minutos y después todo sale de la caché.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.paths import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
FIRST_YEAR = 2023  # primera temporada disponible en OpenF1
MIN_INTERVAL = 2.1  # segundos entre peticiones: plan gratuito = 30/min
MAX_RETRIES = 4
RETRY_BASE = 10  # segundos
TIMEOUT = 90
RETRY_STATUS = {429, 500, 502, 503, 504}
WINDOW_PADDING = timedelta(minutes=10)
OPENF1_ROOT = PROJECT_ROOT / "data" / "openf1"

# Tablas pequeñas: una petición por carrera. La ubicación (x, y) va aparte,
# una petición por piloto, porque es el grueso de los datos.
REPLAY_ENDPOINTS = [
    "drivers",
    "laps",
    "position",
    "intervals",
    "stints",
    "pit",
    "race_control",
]

_last_call = [0.0]

Progress = Callable[[int, int, str], None]


def get_json(endpoint: str, params: dict | None = None) -> list[dict]:
    """GET a OpenF1 respetando el límite de peticiones; 404 = sin resultados."""
    url = f"{BASE_URL}/{endpoint}"
    token = os.environ.get("OPENF1_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            error = e
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return []
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
            error = requests.HTTPError(f"HTTP {resp.status_code}")
        if attempt < MAX_RETRIES:
            backoff = RETRY_BASE * 2**attempt
            logger.warning(
                f"{url}: {error}. Reintento {attempt + 1}/{MAX_RETRIES} en {backoff}s"
            )
            time.sleep(backoff)
    raise ConnectionError(f"No se pudo descargar {url}: {error}")


def to_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def match_session(sessions: list[dict], event_date: date) -> dict | None:
    """Sesión de carrera cuya fecha (UTC) está a ≤ 1 día de la del calendario.

    La tolerancia cubre carreras nocturnas (Las Vegas: sábado local, domingo UTC).
    """
    best, best_diff = None, None
    for session in sessions:
        diff = abs((to_timestamp(session["date_start"]).date() - event_date).days)
        if diff <= 1 and (best_diff is None or diff < best_diff):
            best, best_diff = session, diff
    return best


def find_race_session(year: int, event_date: date) -> dict | None:
    sessions = get_json("sessions", {"year": year, "session_name": "Race"})
    return match_session(sessions, event_date)


def downsample_location(rows: list[dict]) -> pd.DataFrame:
    """Ubicación a 1 Hz (media por segundo), sin los (0, 0) de coche parado."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "x", "y"])
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    df = df[(df["x"] != 0) | (df["y"] != 0)]
    return (
        df.set_index("date")[["x", "y"]]
        .astype(float)
        .resample("1s")
        .mean()
        .dropna()
        .reset_index()
    )


def fastest_lap_window(
    laps: pd.DataFrame, driver_number
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Inicio y fin de la vuelta más rápida de un piloto (para dibujar el trazado)."""
    if laps.empty or "lap_duration" not in laps.columns:
        return None
    own = laps[laps["driver_number"] == driver_number]
    timed = own.dropna(subset=["lap_duration", "date_start"])
    if timed.empty:
        return None
    best = timed.loc[timed["lap_duration"].idxmin()]
    start = to_timestamp(best["date_start"])
    following = own[own["lap_number"] == best["lap_number"] + 1].dropna(
        subset=["date_start"]
    )
    if following.empty:
        end = start + timedelta(seconds=float(best["lap_duration"]))
    else:
        end = to_timestamp(following["date_start"].iloc[0])
    return start, end


def replay_dir(year: int, round_num: int) -> Path:
    return OPENF1_ROOT / f"{year}_R{round_num:02d}"


def is_downloaded(year: int, round_num: int) -> bool:
    return (replay_dir(year, round_num) / "complete").exists()


def _save(df: pd.DataFrame, path: Path) -> None:
    """Parquet no admite columnas con tipos mezclados (p. ej. gaps 1.23 y
    '+1 LAP'): esas columnas se guardan como texto."""
    df = df.copy()
    for col in df.columns[df.dtypes == object]:
        types = {type(v) for v in df[col] if v is not None}
        if len(types) > 1:
            df[col] = df[col].map(lambda v: None if v is None else str(v))
    df.to_parquet(path, index=False)


def _outline(rows: list[dict], window: tuple[pd.Timestamp, pd.Timestamp]):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    start, end = window
    lap = df[(df["date"] >= start) & (df["date"] <= end)]
    lap = lap[(lap["x"] != 0) | (lap["y"] != 0)]
    return lap[["x", "y"]].astype(float).reset_index(drop=True)


def download_race(
    year: int, round_num: int, event_date: date, progress: Progress | None = None
) -> dict:
    """Descarga (o completa) la caché de una carrera y devuelve la sesión.

    Incremental: cada tabla y cada piloto se guardan en cuanto llegan, así que
    una descarga interrumpida se retoma donde se quedó.
    """
    out = replay_dir(year, round_num)
    out.mkdir(parents=True, exist_ok=True)
    session_file = out / "session.json"
    if session_file.exists():
        session = json.loads(session_file.read_text())
    else:
        session = find_race_session(year, event_date)
        if session is None:
            raise LookupError(f"OpenF1 no tiene la carrera del {event_date}")
        session_file.write_text(json.dumps(session))
    key = session["session_key"]

    for i, endpoint in enumerate(REPLAY_ENDPOINTS):
        path = out / f"{endpoint}.parquet"
        if not path.exists():
            if progress:
                progress(i, len(REPLAY_ENDPOINTS), endpoint)
            _save(pd.DataFrame(get_json(endpoint, {"session_key": key})), path)

    drivers = pd.read_parquet(out / "drivers.parquet")
    laps = pd.read_parquet(out / "laps.parquet")
    start = to_timestamp(session["date_start"]) - WINDOW_PADDING
    end_raw = session.get("date_end") or (start + timedelta(hours=4))
    end = to_timestamp(end_raw) + WINDOW_PADDING
    outline_path = out / "outline.parquet"
    numbers = drivers["driver_number"].tolist() if not drivers.empty else []

    for i, number in enumerate(numbers):
        path = out / f"location_{number}.parquet"
        if path.exists():
            continue
        if progress:
            progress(i, len(numbers), f"ubicación del coche {number}")
        rows = get_json(
            "location",
            {
                "session_key": key,
                "driver_number": number,
                "date>": start.isoformat(),
                "date<": end.isoformat(),
            },
        )
        window = fastest_lap_window(laps, number)
        if rows and window and not outline_path.exists():
            outline = _outline(rows, window)
            if len(outline) > 1:
                outline.to_parquet(outline_path, index=False)
        downsample_location(rows).to_parquet(path, index=False)

    (out / "complete").touch()
    logger.info(f"Repetición {year} R{round_num} lista en {out}")
    return session


def load_race(year: int, round_num: int) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Sesión y tablas de una carrera ya descargada."""
    out = replay_dir(year, round_num)
    session = json.loads((out / "session.json").read_text())
    frames = {ep: pd.read_parquet(out / f"{ep}.parquet") for ep in REPLAY_ENDPOINTS}
    locations = []
    for path in sorted(out.glob("location_*.parquet")):
        loc = pd.read_parquet(path)
        loc["driver_number"] = int(path.stem.split("_")[1])
        locations.append(loc)
    frames["location"] = (
        pd.concat(locations, ignore_index=True)
        if locations
        else pd.DataFrame(columns=["date", "x", "y", "driver_number"])
    )
    outline_path = out / "outline.parquet"
    frames["outline"] = (
        pd.read_parquet(outline_path)
        if outline_path.exists()
        else pd.DataFrame(columns=["x", "y"])
    )
    return session, frames
