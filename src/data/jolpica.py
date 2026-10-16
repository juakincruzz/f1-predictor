"""Descarga de resultados de F1 desde la API Jolpica (sucesora de Ergast).

El proyecto solo necesita resultados de carrera y clasificación, y Jolpica los
sirve por temporada en unas pocas llamadas paginadas (frente a cientos con
FastF1). Los parquet se escriben con el mismo esquema que FastF1 para que el
feature engineering no distinga la fuente.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.paths import get_raw_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100
REQUEST_DELAY = 0.5  # Jolpica: 4 req/s en ráfaga, 500 req/h sostenidas
MAX_RETRIES = 4
RETRY_BASE = 5  # segundos
TIMEOUT = 30
RETRY_STATUS = {429, 500, 502, 503, 504}

# constructorId -> TeamName tal y como lo escribe FastF1, para que el ritmo de
# equipo no se corte al mezclar fuentes ("Red Bull" vs "Red Bull Racing")
TEAM_NAMES = {
    "red_bull": "Red Bull Racing",
    "alpine": "Alpine",
    "alfa": "Alfa Romeo",
    "rb": "RB",
    "sauber": "Kick Sauber",
    "cadillac": "Cadillac",
}

TEAM_COLORS = {
    "red_bull": "3671C6",
    "ferrari": "E8002D",
    "mercedes": "27F4D2",
    "mclaren": "FF8000",
    "aston_martin": "229971",
    "alpine": "FF87BC",
    "williams": "64C4FF",
    "rb": "6692FF",
    "alphatauri": "5E8FAA",
    "toro_rosso": "469BFF",
    "sauber": "52E252",
    "alfa": "C92D4B",
    "haas": "B6BABD",
    "audi": "F50537",
    "cadillac": "AAAAAD",
    "racing_point": "F596C8",
    "force_india": "F596C8",
    "renault": "FFF500",
}
DEFAULT_TEAM_COLOR = "C0C0C0"

SESSION_KEYS = {"R": "Results", "Q": "QualifyingResults"}
SESSION_ENDPOINTS = {"R": "results", "Q": "qualifying"}


def _get(path: str, params: dict | None = None) -> dict:
    """GET a Jolpica con reintentos y backoff ante 429/5xx o fallos de red."""
    url = f"{BASE_URL}/{path}"
    error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as e:
            error = e
        else:
            if resp.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return resp.json()
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
            error = requests.HTTPError(f"HTTP {resp.status_code}")
        if attempt < MAX_RETRIES:
            wait = RETRY_BASE * 2**attempt
            logger.warning(
                f"{url}: {error}. Reintento {attempt + 1}/{MAX_RETRIES} en {wait}s..."
            )
            time.sleep(wait)
    raise ConnectionError(f"No se pudo descargar {url}: {error}")


def _merge_race(races: dict[str, dict], race: dict) -> None:
    """Fusiona una carrera en el acumulado; la paginación es por filas, así que
    una misma carrera puede llegar repartida entre dos páginas."""
    key = race["round"]
    if key not in races:
        races[key] = {k: list(v) if isinstance(v, list) else v for k, v in race.items()}
        return
    for list_key in SESSION_KEYS.values():
        if list_key in race:
            races[key].setdefault(list_key, []).extend(race[list_key])


def fetch_races(year: int, endpoint: str = "") -> list[dict]:
    """Descarga todas las páginas de un endpoint de temporada.

    endpoint "" = calendario, "results" = carreras, "qualifying" = clasificación.
    """
    path = f"{year}/{endpoint}.json" if endpoint else f"{year}.json"
    races: dict[str, dict] = {}
    offset = 0
    while True:
        data = _get(path, {"limit": PAGE_LIMIT, "offset": offset})["MRData"]
        for race in data["RaceTable"]["Races"]:
            _merge_race(races, race)
        offset += PAGE_LIMIT
        if offset >= int(data["total"]):
            break
    return sorted(races.values(), key=lambda r: int(r["round"]))


def parse_schedule(races: list[dict], year: int) -> pd.DataFrame:
    """Calendario de la temporada: ronda, nombre del GP, fecha y lugar."""
    rows = [
        {
            "Year": year,
            "RoundNumber": int(race["round"]),
            "EventName": race["raceName"],
            "EventDate": race["date"],
            "Country": race["Circuit"]["Location"].get("country", ""),
            "Locality": race["Circuit"]["Location"].get("locality", ""),
        }
        for race in races
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Year",
            "RoundNumber",
            "EventName",
            "EventDate",
            "Country",
            "Locality",
        ],
    )
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    return df


def _driver_fields(entry: dict) -> dict:
    driver = entry["Driver"]
    constructor = entry["Constructor"]
    team_id = constructor["constructorId"]
    given = driver.get("givenName", "")
    family = driver.get("familyName", "")
    return {
        "DriverNumber": str(entry.get("number") or driver.get("permanentNumber", "")),
        "Abbreviation": driver.get("code") or family[:3].upper(),
        "DriverId": driver["driverId"],
        "FirstName": given,
        "LastName": family,
        "FullName": f"{given} {family}".strip(),
        "TeamName": TEAM_NAMES.get(team_id, constructor["name"]),
        "TeamId": team_id,
        "TeamColor": TEAM_COLORS.get(team_id, DEFAULT_TEAM_COLOR),
    }


def parse_session(races: list[dict], year: int, session_type: str) -> pd.DataFrame:
    """Convierte resultados Jolpica (R o Q) al esquema de session.results de FastF1."""
    rows = []
    for race in races:
        entries = race.get(SESSION_KEYS[session_type], [])
        for entry in entries:
            row = _driver_fields(entry)
            row["Position"] = float(entry["position"])
            if session_type == "R":
                grid = float(entry.get("grid", "nan"))
                # grid 0 = sale desde el pit lane: cuenta como última posición
                row["GridPosition"] = float(len(entries)) if grid == 0 else grid
                row["ClassifiedPosition"] = entry.get("positionText", "")
                row["Status"] = entry.get("status", "")
                row["Points"] = float(entry.get("points", 0))
            row["Year"] = year
            row["RoundNumber"] = int(race["round"])
            row["SessionType"] = session_type
            row["EventName"] = race["raceName"]
            rows.append(row)
    return pd.DataFrame(rows)


def collect_year_jolpica(year: int, force: bool = False) -> dict:
    """Descarga calendario, carreras y clasificaciones de una temporada.

    Incremental: solo escribe las rondas cuyo parquet falta (salvo force). Las
    rondas aún sin disputar cuentan como 'pending'.
    """
    logger.info(f"Jolpica: temporada {year}...")
    year_dir = get_raw_path(str(year))
    schedule = parse_schedule(fetch_races(year), year)
    stats = {"processed": 0, "skipped": 0, "pending": 0, "total": 2 * len(schedule)}
    if schedule.empty:
        logger.warning(f"Sin calendario para {year}")
        return stats
    schedule.to_parquet(year_dir / "schedule.parquet", index=False)

    rounds = schedule["RoundNumber"].tolist()
    for session_type, endpoint in SESSION_ENDPOINTS.items():
        missing = [
            r
            for r in rounds
            if force or not (year_dir / f"R{r:02d}_{session_type}.parquet").exists()
        ]
        stats["skipped"] += len(rounds) - len(missing)
        if not missing:
            continue
        df = parse_session(fetch_races(year, endpoint), year, session_type)
        written = set()
        if not df.empty:
            for round_num, group in df.groupby("RoundNumber"):
                if round_num in missing:
                    path = year_dir / f"R{round_num:02d}_{session_type}.parquet"
                    group = group.assign(TotalRounds=len(schedule))
                    group.reset_index(drop=True).to_parquet(path, index=False)
                    written.add(round_num)
        stats["processed"] += len(written)
        stats["pending"] += len(missing) - len(written)

    logger.info(
        f"Temporada {year}: procesados {stats['processed']}, saltados "
        f"{stats['skipped']}, pendientes {stats['pending']}"
    )
    return stats
