# Rediseño rojo/plata y cierre del proyecto — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dashboard con estética F1 clásica (rojo Fórmula + plata), predicción de la próxima carrera real con datos 2024–2026 descargados de Jolpica, y repo limpio.

**Architecture:** Nuevo colector `src/data/jolpica.py` que escribe parquets con el esquema FastF1 (el feature engineering no cambia de forma). `src/models/upcoming.py` añade la carrera futura al histórico sin resultado y reutiliza `build_features_from_frames` (las features con `shift(1)` no ven la carrera). `src/models/explain.py` calcula importancia y SHAP nativo de XGBoost. El dashboard se divide en `theme.py` (CSS/paleta), `components.py` (HTML) y `app.py`.

**Tech Stack:** Python 3.14 (venv) / 3.11 (CI), pandas 2.3, XGBoost 3.3, Streamlit 1.59, Altair 6, requests, pytest, ruff, black.

**Spec:** `docs/superpowers/specs/2026-10-16-redesign-finish-design.md`

## Global Constraints

- Clases fijas: `0 = No Podium, 1 = P1, 2 = P2, 3 = P3`; `model.num_class: 4`.
- Módulos bajo `src/` insertan la raíz en `sys.path` y usan imports absolutos (`from src...`).
- Rutas siempre vía `src/utils/paths.py`; hiperparámetros/rangos en `config.yaml`.
- Parquets crudos: `data/raw/{year}/R{nn}_{R|Q}.parquet`, esquema FastF1; calendario en `data/raw/{year}/schedule.parquet`.
- Paleta: rojo `#E10600`, rojo oscuro `#A30500`, plata `#C9CDD2`, plata clara `#EEF0F2`, plata oscura `#8A8F96`, carbono `#0B0B0D`, panel `#17181C`, línea `#2A2C31`, texto `#D9D9DE`. Fuente Titillium Web.
- Tests sin red. `ruff check src tests`, `black --check src tests`, `pytest` verdes.
- Comandos con `venv/bin/python -m ...` en local.

## Review Focus

1. Jolpica parte una misma carrera entre dos páginas (paginación por filas) → los resultados deben fusionarse, no duplicar ni perder pilotos. (Task 1)
2. Novato sin historial de qualy en la próxima carrera → parrilla estimada al final, sin crash. (Task 4)
3. Temporada terminada / sin calendario / ronda cancelada sin resultados → `next_race` devuelve `None` o salta la ronda; el dashboard muestra un aviso. (Task 4, Task 6)
4. Piloto que cambia de dorsal entre temporadas → su forma reciente continúa (clave `DriverId`). (Task 2)
5. Dashboard sin datos ni modelo (CI) → renderiza avisos sin excepción. (Task 6)

---

### Task 1: Colector Jolpica

**Files:**
- Create: `src/data/jolpica.py`
- Test: `tests/test_jolpica.py`

**Interfaces:**
- Produces: `fetch_races(year: int, endpoint: str = "") -> list[dict]`, `parse_schedule(races: list[dict], year: int) -> pd.DataFrame` (cols `Year, RoundNumber, EventName, EventDate, Country, Locality`), `parse_session(races: list[dict], year: int, session_type: str) -> pd.DataFrame` (esquema FastF1), `collect_year_jolpica(year: int, force: bool = False) -> dict` (`processed, skipped, pending, total`), `TEAM_NAMES`, `TEAM_COLORS`, `DEFAULT_TEAM_COLOR`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests del colector Jolpica (sin red: se parchea _get / fetch_races)."""

import pandas as pd

from src.data import jolpica


def _entry(number, code, driver_id, team_id, team_name, position, grid="1"):
    return {
        "number": number,
        "position": str(position),
        "positionText": str(position),
        "points": "0",
        "grid": grid,
        "status": "Finished",
        "Driver": {
            "driverId": driver_id,
            "code": code,
            "givenName": "Given",
            "familyName": driver_id.capitalize(),
        },
        "Constructor": {"constructorId": team_id, "name": team_name},
    }


def _race(rnd, results, key="Results"):
    return {
        "season": "2026",
        "round": str(rnd),
        "raceName": f"GP {rnd}",
        "date": f"2026-03-0{rnd}",
        "Circuit": {"Location": {"country": "X", "locality": "Y"}},
        key: results,
    }


def test_fetch_races_merges_race_split_across_pages(monkeypatch):
    pages = [
        {"MRData": {"total": "3", "RaceTable": {"Races": [
            _race(1, [_entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)]),
        ]}}},
        {"MRData": {"total": "3", "RaceTable": {"Races": [
            _race(1, [_entry("4", "NOR", "norris", "mclaren", "McLaren", 2)]),
            _race(2, [_entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)]),
        ]}}},
    ]
    calls = []

    def fake_get(path, params=None):
        calls.append(params["offset"])
        return pages[len(calls) - 1]

    monkeypatch.setattr(jolpica, "PAGE_LIMIT", 2)
    monkeypatch.setattr(jolpica, "_get", fake_get)

    races = jolpica.fetch_races(2026, "results")

    assert calls == [0, 2]
    assert [r["round"] for r in races] == ["1", "2"]
    assert len(races[0]["Results"]) == 2


def test_parse_session_race_uses_fastf1_schema_and_team_names():
    races = [_race(3, [
        _entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1, grid="2"),
        _entry("30", None, "lawson", "rb", "RB F1 Team", 2, grid="0"),
    ])]
    df = jolpica.parse_session(races, 2026, "R")

    assert df["TeamName"].tolist() == ["Red Bull Racing", "RB"]
    assert df["Abbreviation"].tolist() == ["VER", "LAW"]
    assert df["DriverNumber"].tolist() == ["1", "30"]
    assert df["Position"].tolist() == [1.0, 2.0]
    # Salida desde el pit lane (grid 0) cuenta como última posición
    assert df["GridPosition"].tolist() == [2.0, 2.0]
    assert df["RoundNumber"].unique().tolist() == [3]
    assert df["SessionType"].unique().tolist() == ["R"]
    assert df["TeamColor"].iloc[0] == jolpica.TEAM_COLORS["red_bull"]


def test_parse_session_qualifying_has_no_grid_column():
    races = [_race(1, [_entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)],
                   key="QualifyingResults")]
    df = jolpica.parse_session(races, 2026, "Q")
    assert df["Position"].tolist() == [1.0]
    assert "GridPosition" not in df.columns


def test_parse_schedule():
    sched = jolpica.parse_schedule([_race(1, []), _race(2, [])], 2026)
    assert sched["RoundNumber"].tolist() == [1, 2]
    assert sched["EventName"].tolist() == ["GP 1", "GP 2"]
    assert sched["Year"].unique().tolist() == [2026]
    assert pd.api.types.is_datetime64_any_dtype(sched["EventDate"])


def test_collect_year_jolpica_writes_and_is_incremental(monkeypatch, tmp_path):
    def fake_fetch(year, endpoint=""):
        if endpoint == "":
            return [_race(1, []), _race(2, [])]
        key = "Results" if endpoint == "results" else "QualifyingResults"
        # Solo la ronda 1 se ha disputado
        return [_race(1, [_entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)], key)]

    monkeypatch.setattr(jolpica, "fetch_races", fake_fetch)
    monkeypatch.setattr(jolpica, "get_raw_path", lambda sub=None: tmp_path)

    stats = jolpica.collect_year_jolpica(2026)
    assert (tmp_path / "schedule.parquet").exists()
    assert (tmp_path / "R01_R.parquet").exists()
    assert (tmp_path / "R01_Q.parquet").exists()
    assert not (tmp_path / "R02_R.parquet").exists()
    assert stats == {"processed": 2, "skipped": 0, "pending": 2, "total": 4}

    stats = jolpica.collect_year_jolpica(2026)
    assert stats == {"processed": 0, "skipped": 2, "pending": 2, "total": 4}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_jolpica.py -v`
Expected: FAIL — `ImportError: cannot import name 'jolpica'`

- [ ] **Step 3: Implement `src/data/jolpica.py`**

```python
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
        columns=["Year", "RoundNumber", "EventName", "EventDate", "Country", "Locality"],
    )
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    return df


def _driver_fields(entry: dict) -> dict:
    driver = entry["Driver"]
    constructor = entry["Constructor"]
    team_id = constructor["constructorId"]
    return {
        "DriverNumber": str(entry.get("number") or driver.get("permanentNumber", "")),
        "Abbreviation": driver.get("code") or driver["familyName"][:3].upper(),
        "DriverId": driver["driverId"],
        "FirstName": driver.get("givenName", ""),
        "LastName": driver.get("familyName", ""),
        "FullName": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
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

    for session_type, endpoint in SESSION_ENDPOINTS.items():
        rounds = schedule["RoundNumber"].tolist()
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
                    group.reset_index(drop=True).to_parquet(path, index=False)
                    written.add(round_num)
        stats["processed"] += len(written)
        stats["pending"] += len(missing) - len(written)

    logger.info(
        f"Temporada {year}: procesados {stats['processed']}, saltados "
        f"{stats['skipped']}, pendientes {stats['pending']}"
    )
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_jolpica.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/jolpica.py tests/test_jolpica.py
git commit -m "feat: colector Jolpica con esquema FastF1"
```

---

### Task 2: CLI de descarga, calendario y clave estable de piloto

**Files:**
- Modify: `src/data/collect.py` (añadir `load_schedule`, `run`, `main`)
- Modify: `scripts/download_years.py` (reescribir como wrapper con argumentos)
- Modify: `src/features/build_features.py` (`driver_key`, conservar `DriverId/FullName/TeamColor`)
- Modify: `Makefile` (target `update`)
- Test: `tests/test_features.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `collect_year_jolpica(year, force)` (Task 1).
- Produces: `load_schedule(year: int) -> pd.DataFrame`, `run(years: list[int], source: str = "jolpica", force: bool = False) -> None`, `driver_key(df: pd.DataFrame) -> pd.Series`; `build_features_from_frames` además conserva `DriverId`, `FullName`, `TeamColor` si existen.

- [ ] **Step 1: Write the failing tests** (añadir a `tests/test_features.py`)

```python
from src.features.build_features import compute_races_experience, driver_key


def test_recent_form_follows_driver_id_across_number_change():
    """Un piloto que cambia de dorsal mantiene su forma reciente."""
    df = pd.DataFrame(
        {
            "Year": [2025, 2026],
            "RoundNumber": [24, 1],
            "DriverNumber": ["1", "3"],
            "DriverId": ["max_verstappen", "max_verstappen"],
            "final_position": [2, 5],
        }
    )
    result = compute_recent_form(df, n=5)
    assert result.loc[result["Year"] == 2026, "recent_form_avg"].iloc[0] == 2
    exp = compute_races_experience(df)
    assert exp.loc[exp["Year"] == 2026, "races_experience"].iloc[0] == 1


def test_driver_key_falls_back_to_number():
    df = pd.DataFrame({"DriverNumber": ["44", "1"], "DriverId": ["hamilton", None]})
    assert driver_key(df).tolist() == ["hamilton", "1"]
    assert driver_key(df[["DriverNumber"]]).tolist() == ["44", "1"]
```

Y en `tests/test_data.py`:

```python
from src.data.collect import load_schedule


def test_load_schedule_missing_returns_empty():
    assert load_schedule(1900).empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_features.py tests/test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'driver_key'` / `'load_schedule'`

- [ ] **Step 3: Implement**

`src/features/build_features.py` — añadir tras `build_targets` y usarlo en `compute_recent_form`, `compute_track_history`, `compute_races_experience`:

```python
def driver_key(df: pd.DataFrame) -> pd.Series:
    """Identificador estable del piloto: DriverId (el dorsal puede cambiar entre
    temporadas y repetirse entre épocas), con fallback a DriverNumber."""
    if "DriverId" in df.columns:
        ids = df["DriverId"].replace("", pd.NA)
        return ids.fillna(df["DriverNumber"])
    return df["DriverNumber"]
```

```python
# compute_recent_form
    df["recent_form_avg"] = _rolling_prev_mean(
        df.groupby(driver_key(df))["final_position"], n
    )
# compute_track_history
    df["track_history_avg"] = _rolling_prev_mean(
        df.groupby([driver_key(df), df["EventName"]])["final_position"], n
    )
# compute_races_experience
    df["races_experience"] = df.groupby(driver_key(df)).cumcount()
```

En `build_features_from_frames`, `base_cols` pasa a:

```python
    base_cols = [
        "Year",
        "RoundNumber",
        "EventName",
        "DriverNumber",
        "DriverId",
        "Abbreviation",
        "FullName",
        "TeamName",
        "TeamColor",
        "target",
    ]
```

`src/data/collect.py` — añadir `import argparse`, `from src.data.jolpica import collect_year_jolpica` y al final (sustituyendo el `__main__`):

```python
def load_schedule(year: int) -> pd.DataFrame:
    """Calendario guardado por el colector Jolpica (vacío si no existe)."""
    path = get_raw_path(str(year)) / "schedule.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def run(years: list[int], source: str = "jolpica", force: bool = False) -> None:
    """Descarga incremental de las temporadas indicadas con la fuente elegida."""
    for year in years:
        if source == "jolpica":
            collect_year_jolpica(year, force=force)
        else:
            collect_year(year, force=force)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Descarga incremental de datos F1.")
    parser.add_argument("--source", choices=["jolpica", "fastf1"], default="jolpica")
    parser.add_argument(
        "--years", nargs="+", type=int, help="Temporadas (defecto: rango de config)"
    )
    parser.add_argument("--force", action="store_true", help="Re-descarga todo")
    args = parser.parse_args(argv)
    years = args.years or list(
        range(config.get("data.start_year"), config.get("data.end_year") + 1)
    )
    run(years, source=args.source, force=args.force)


if __name__ == "__main__":
    main()
```

`scripts/download_years.py`:

```python
#!/usr/bin/env python3
"""Descarga temporadas concretas; pensado para ejecutarse en background.

Uso:
    nohup python3 scripts/download_years.py 2024 2025 2026 > data/download.log 2>&1 &
    python3 scripts/download_years.py 2019 --source fastf1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="+", type=int)
    parser.add_argument("--source", choices=["jolpica", "fastf1"], default="jolpica")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.years, source=args.source, force=args.force)
```

`Makefile` — añadir `update` a `.PHONY`, a `help` y:

```make
update:
	python -m src.data.collect --years $$(date +%Y)
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests -v`
Expected: todo PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/collect.py src/features/build_features.py scripts/download_years.py Makefile tests/
git commit -m "feat: CLI de descarga multi-fuente, calendario y clave DriverId"
```

---

### Task 3: Entrenamiento con refit completo y features explícitas

**Files:**
- Modify: `src/models/train.py`
- Modify: `config.yaml` (`validation.val_year: 2025`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `train_model(df=None, test_size=None, refit_full: bool = False)` (misma tupla de retorno); `prepare_xy` selecciona solo `features.columns` presentes en el df (columnas extra como `FullName` no entran al modelo).

- [ ] **Step 1: Write the failing tests** (añadir a `tests/test_models.py`)

```python
def _synthetic_years(years, n_per_year=60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = n_per_year * len(years)
    grid = rng.integers(1, 21, n).astype(float)
    return pd.DataFrame(
        {
            "Year": np.repeat(years, n_per_year),
            "RoundNumber": np.tile(np.arange(n_per_year) // 20 + 1, len(years)),
            "DriverNumber": [str(i % 20) for i in range(n)],
            "FullName": ["Driver"] * n,
            "target": np.where(grid <= 3, grid, 0).astype(int),
            "grid_position": grid,
            "recent_form_avg": rng.uniform(1, 20, n),
        }
    )


def test_prepare_xy_ignores_non_feature_columns():
    X, y = prepare_xy(_synthetic_years([2024]))
    assert list(X.columns) == ["grid_position", "recent_form_avg"]
    assert len(y) == 60


def test_refit_full_retrains_on_all_years_with_best_iteration():
    df = _synthetic_years([2023, 2024, 2025])  # val_year = 2025 en config
    es_model, X_train, *_ = train_model(df, test_size=0.2)
    assert len(X_train) == 120  # solo años < 2025
    full_model, *_ = train_model(df, test_size=0.2, refit_full=True)
    assert full_model.get_booster().num_boosted_rounds() == es_model.best_iteration + 1
```

(importar `prepare_xy` junto a `train_model`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — columnas extra en X / `unexpected keyword argument 'refit_full'`

- [ ] **Step 3: Implement**

`config.yaml`: `val_year: 2025`.

`src/models/train.py`: sustituir `DROP_COLS` por:

```python
FEATURE_COLUMNS = config.get("features.columns", [])


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Solo las columnas de features declaradas en config.yaml."""
    return df[[c for c in FEATURE_COLUMNS if c in df.columns]]
```

```python
def prepare_xy(df: pd.DataFrame, medians: pd.Series | None = None):
    """Separa X (features de config.yaml) e y.

    Los NaN se rellenan con `medians` (las del train para evitar leakage);
    si no se pasan, se usan las del propio df como fallback.
    """
    X = _feature_frame(df)
    y = df["target"].astype(int)
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X, y


def _new_model(**overrides) -> XGBClassifier:
    params = {**config.get("model.params", {}), **overrides}
    return XGBClassifier(
        objective=config.get("model.objective", "multi:softprob"),
        num_class=config.get("model.num_class", 4),
        eval_metric=config.get("model.eval_metric", "mlogloss"),
        **params,
    )
```

En `train_model`: firma `(df=None, test_size=None, refit_full: bool = False)`, docstring añade "Con refit_full, tras el early stopping reentrena con TODO el df usando best_iteration + 1 árboles (el modelo desplegado conoce las temporadas recientes).", `train_medians = _feature_frame(train_df).median(numeric_only=True)`, `model = _new_model()`, y antes del `return`:

```python
    if refit_full and val_df is not None and best_iteration is not None:
        n_trees = best_iteration + 1
        X_full, y_full = prepare_xy(df)
        model = _new_model(n_estimators=n_trees)
        model.fit(X_full, y_full, verbose=False)
        logger.info(
            f"Reentrenado con todas las temporadas ({len(df)} filas, {n_trees} árboles)."
        )
```

`__main__`: `model, *_ = train_model(df, refit_full=True)`.

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests -v` → todo PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/train.py config.yaml tests/test_models.py
git commit -m "feat: refit con todas las temporadas tras early stopping"
```

---

### Task 4: Predicción de la próxima carrera

**Files:**
- Create: `src/models/upcoming.py`
- Test: `tests/test_upcoming.py`

**Interfaces:**
- Consumes: `load_schedule` (Task 2), `driver_key`, `build_features_from_frames`, `load_raw_range`, `predict_for_race`.
- Produces: `UpcomingRace(year, round_num, event_name, event_date: date, total_rounds)`, `UpcomingPrediction(race, grid_source, predictions: pd.DataFrame)`, `GRID_REAL = "real"`, `GRID_ESTIMATED = "estimada"`, `next_race(schedule, completed_rounds: set[int], today: date) -> UpcomingRace | None`, `build_upcoming_entries(race_hist, quali_hist, race, n_recent=5) -> tuple[pd.DataFrame, str]`, `predict_next_race(today: date | None = None, model_path="xgb_model.json") -> UpcomingPrediction | None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests de la predicción de la próxima carrera (sin red ni modelo)."""

from datetime import date

import pandas as pd

from src.features.build_features import build_features_from_frames
from src.models.upcoming import (
    GRID_ESTIMATED,
    GRID_REAL,
    UpcomingRace,
    build_upcoming_entries,
    next_race,
)

SCHEDULE = pd.DataFrame(
    {
        "Year": [2026] * 4,
        "RoundNumber": [1, 2, 3, 4],
        "EventName": ["GP A", "GP B", "GP C", "GP D"],
        "EventDate": pd.to_datetime(
            ["2026-03-01", "2026-03-15", "2026-04-01", "2026-04-15"]
        ),
    }
)
RACE = UpcomingRace(2026, 3, "GP C", date(2026, 4, 1), 4)


def _session(rounds, drivers, session_type):
    rows = []
    for rnd in rounds:
        for pos, (num, did, team) in enumerate(drivers, 1):
            rows.append(
                {
                    "Year": 2026,
                    "RoundNumber": rnd,
                    "EventName": f"GP {'ABCD'[rnd - 1]}",
                    "DriverNumber": num,
                    "DriverId": did,
                    "Abbreviation": did[:3].upper(),
                    "FullName": did,
                    "TeamName": team,
                    "TeamColor": "C0C0C0",
                    "Position": float(pos),
                    "GridPosition": float(pos),
                    "SessionType": session_type,
                }
            )
    return pd.DataFrame(rows)


DRIVERS = [("1", "verstappen", "Red Bull Racing"), ("4", "norris", "McLaren")]


def test_next_race_returns_first_pending_round():
    race = next_race(SCHEDULE, {1, 2}, date(2026, 3, 20))
    assert (race.round_num, race.event_name, race.total_rounds) == (3, "GP C", 4)
    assert race.event_date == date(2026, 4, 1)


def test_next_race_skips_stale_round_without_results():
    # La ronda 2 pasó hace semanas sin resultados (cancelada): se salta
    race = next_race(SCHEDULE, {1}, date(2026, 3, 30))
    assert race.round_num == 3


def test_next_race_none_when_season_over_or_no_schedule():
    assert next_race(SCHEDULE, {1, 2, 3, 4}, date(2026, 5, 1)) is None
    assert next_race(pd.DataFrame(), set(), date(2026, 5, 1)) is None


def test_entries_use_real_qualifying_when_available():
    race_hist = _session([1, 2], DRIVERS, "R")
    quali = _session([1, 2, 3], list(reversed(DRIVERS)), "Q")
    entries, source = build_upcoming_entries(race_hist, quali, RACE)
    assert source == GRID_REAL
    grid = dict(zip(entries["DriverId"], entries["GridPosition"]))
    assert grid == {"norris": 1.0, "verstappen": 2.0}
    assert entries["Position"].isna().all()
    assert set(entries["RoundNumber"]) == {3}


def test_entries_estimate_grid_and_put_rookie_last():
    race_hist = _session([2], DRIVERS + [("99", "rookie", "Cadillac")], "R")
    quali = _session([1, 2], list(reversed(DRIVERS)), "Q")
    entries, source = build_upcoming_entries(race_hist, quali, RACE)
    assert source == GRID_ESTIMATED
    grid = dict(zip(entries["DriverId"], entries["GridPosition"]))
    assert grid == {"norris": 1.0, "verstappen": 2.0, "rookie": 3.0}


def test_upcoming_features_only_use_past_races():
    race_hist = _session([1, 2], DRIVERS, "R")
    entries, _ = build_upcoming_entries(race_hist, pd.DataFrame(), RACE)
    feats = build_features_from_frames(
        pd.concat([race_hist, entries], ignore_index=True), pd.DataFrame()
    )
    upcoming = feats[feats["RoundNumber"] == 3].set_index("DriverId")
    assert upcoming.loc["verstappen", "recent_form_avg"] == 1.0
    assert upcoming.loc["norris", "recent_form_avg"] == 2.0
    assert upcoming.loc["verstappen", "races_experience"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_upcoming.py -v`
Expected: FAIL — `ModuleNotFoundError: src.models.upcoming`

- [ ] **Step 3: Implement `src/models/upcoming.py`**

```python
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
        recent = positions.groupby(driver_key(q)).agg(
            lambda s: s.tail(n_recent).mean()
        )
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
        print(f"{r.year} R{r.round_num} {r.event_name} ({r.event_date}) — parrilla {result.grid_source}")
        print(result.predictions[["Abbreviation", "TeamName", "grid_position", "PodiumProbability"]].head(10))
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests -v` → todo PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/upcoming.py tests/test_upcoming.py
git commit -m "feat: predicción de la próxima carrera con parrilla real o estimada"
```

---

### Task 5: Explicabilidad (importancia + SHAP nativo)

**Files:**
- Create: `src/models/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Produces: `FEATURE_LABELS: dict[str, str]`, `feature_importance(model) -> pd.DataFrame` (cols `feature, label, importance`, suma 1), `podium_shap(model, X: pd.DataFrame) -> pd.DataFrame` (cols `feature, label, shap`, media |SHAP| sumada sobre P1–P3).

- [ ] **Step 1: Write the failing test**

```python
"""Tests de explicabilidad del modelo."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.explain import feature_importance, podium_shap


def _model_and_x():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"grid_position": rng.uniform(1, 20, 200), "race_round": rng.uniform(0, 1, 200)}
    )
    y = np.select([X.grid_position < 2, X.grid_position < 3, X.grid_position < 4], [1, 2, 3], 0)
    model = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=20)
    model.fit(X, y)
    return model, X


def test_feature_importance_normalized_and_labeled():
    model, _ = _model_and_x()
    imp = feature_importance(model)
    assert set(imp["feature"]) == {"grid_position", "race_round"}
    assert abs(imp["importance"].sum() - 1) < 1e-9
    assert imp.iloc[0]["feature"] == "grid_position"
    assert imp.iloc[0]["label"] == "Posición de salida"


def test_podium_shap_ranks_grid_first():
    model, X = _model_and_x()
    shap_df = podium_shap(model, X.head(20))
    assert list(shap_df.columns) == ["feature", "label", "shap"]
    assert shap_df.iloc[0]["feature"] == "grid_position"
    assert (shap_df["shap"] >= 0).all()
```

- [ ] **Step 2: Run to verify fail** — `venv/bin/python -m pytest tests/test_explain.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/models/explain.py`**

```python
"""Explicabilidad del modelo: importancia global y SHAP por carrera.

Usa los valores SHAP nativos de XGBoost (pred_contribs, TreeSHAP exacto), que
no dependen de la compatibilidad de la librería shap con cada versión.
"""

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

PODIUM_CLASSES = [1, 2, 3]
FEATURE_LABELS = {
    "grid_position": "Posición de salida",
    "recent_form_avg": "Forma reciente",
    "team_pace_avg": "Ritmo del equipo",
    "track_history_avg": "Historial en el circuito",
    "championship_position": "Posición en el campeonato",
    "races_experience": "Experiencia (carreras)",
    "teammate_gap": "Gap con el compañero",
    "race_round": "Momento de la temporada",
}


def _labeled(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    return (
        df[["feature", "label", value_col]]
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def feature_importance(model: XGBClassifier) -> pd.DataFrame:
    """Importancia por ganancia (gain) normalizada a 1."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    names = booster.feature_names or list(scores)
    df = pd.DataFrame(
        {"feature": names, "importance": [scores.get(n, 0.0) for n in names]}
    )
    total = df["importance"].sum()
    if total > 0:
        df["importance"] = df["importance"] / total
    return _labeled(df, "importance")


def podium_shap(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Media de |SHAP| por feature, sumada sobre las clases de podio."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    # (filas, clases, features + sesgo) -> descarta el sesgo
    podium = abs(contribs[:, PODIUM_CLASSES, :-1]).sum(axis=1)
    df = pd.DataFrame({"feature": list(X.columns), "shap": podium.mean(axis=0)})
    return _labeled(df, "shap")
```

- [ ] **Step 4: Run tests** → todo PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/explain.py tests/test_explain.py
git commit -m "feat: importancia de features y SHAP nativo por carrera"
```

---

### Task 6: Dashboard rojo Fórmula / plata

**Files:**
- Create: `.streamlit/config.toml`, `src/dashboard/theme.py`, `src/dashboard/components.py`
- Modify (reescribir): `src/dashboard/app.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `predict_next_race`, `GRID_REAL`, `UpcomingPrediction` (Task 4); `feature_importance`, `podium_shap` (Task 5); `collect_year_jolpica` (Task 1); `build_features_for_year`, `predict_for_race`, `prepare_xy`, `load_model`.
- Produces: `theme.apply_theme()`, `theme.style_chart(chart)`, constantes de color; `components.header/section/stat_cards/podium_cards/prediction_table/model_missing`.

- [ ] **Step 1: Write the failing test**

```python
"""Smoke test del dashboard: debe renderizar sin excepciones con o sin datos."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
```

- [ ] **Step 2: Run to verify fail** — `venv/bin/python -m pytest tests/test_dashboard.py -v` → falla (`len(at.tabs) == 4` vale, pero el rediseño aún no existe: el test se escribe contra la app nueva; si la vieja ya pasa, continuar).

- [ ] **Step 3: Implement**

`.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#E10600"
backgroundColor = "#0B0B0D"
secondaryBackgroundColor = "#17181C"
textColor = "#D9D9DE"

[browser]
gatherUsageStats = false
```

`src/dashboard/theme.py`: constantes de la paleta (Global Constraints), `CSS` con: import Google Fonts Titillium Web; fuente global; fondo `.stApp` radial carbono; sidebar con franja roja superior; `.f1-header` plata cepillada (`repeating-linear-gradient` + degradado cromado), borde inferior rojo 5px y franjas rojas inclinadas en `::after` (ocultas < 640px); `.f1-title` 900 itálica mayúsculas con `span` en rojo; `.f1-section` con barra roja inclinada y línea plata; `.f1-stats/.f1-stat` (borde superior plata, `.accent` rojo); `.f1-podium` grid 3 columnas con orden P2-P1-P3 y alturas 200/230/180 (1 columna y orden P1-P2-P3 < 640px); `.f1-card` con franja de color de equipo (`--team`), `.f1-pos` badge sesgado (P1 rojo, P2/P3 cromado), `.f1-abbr`, `.f1-name`, `.f1-team`, `.f1-bar`; tabs en mayúsculas con highlight rojo; botones cromados con hover rojo; `[data-testid="stDecoration"]` degradado rojo→plata. `apply_theme()` lo inyecta con `st.markdown(..., unsafe_allow_html=True)`. `style_chart(chart)` aplica fondo transparente, fuente, ejes plata, grid `#2A2C31`, leyenda arriba.

`src/dashboard/components.py`: helpers HTML (todo texto con `html.escape`, color de equipo validado `^[0-9A-Fa-f]{6}$` con fallback plata):
- `header(main, accent, subtitle)`, `section(title)`, `stat_cards(items: list[tuple[str, str]])`, `model_missing()`
- `podium_cards(df, prob_label="Prob. podio")`: top-3 del df en orden; cada tarjeta P1/P2/P3 con abreviatura, nombre, equipo y barra de `PodiumProbability`.
- `prediction_table(preds, with_actual=False)`: `st.dataframe` con Pos, Piloto, Equipo, Salida, P1/P2/P3 (%), Podio (`ProgressColumn` 0–100) y Real (P1/P2/P3/—) si `with_actual`.

`src/dashboard/app.py`: `set_page_config` + `apply_theme()` + `header`; caches `season_features(year)`, `race_predictions(year, round, mtime)`, `upcoming(today, mtime)` (`st.cache_data`) y `model(mtime)` (`st.cache_resource`); sidebar: temporada (años con `*_R.parquet`, la más reciente por defecto), Gran Premio (`R05 · Monaco Grand Prix`), botón «Actualizar temporada actual» → `collect_year_jolpica(today.year)` + `st.cache_data.clear()` + toast vía `session_state` + `st.rerun()`, errores con `st.error`; tabs:
- 🏁 Próxima carrera: stat cards (GP, fecha, ronda x/total, parrilla real/estimada), aviso si estimada, podio previsto, tabla.
- 🔮 Predicción: stat cards (carrera, aciertos x/3, ganador real, favorito), podio previsto y podio real, tabla con Real, caption sobre in-sample.
- 📊 Histórico: stat cards con medias, Altair líneas Accuracy (plata) / F1 macro (rojo), tabla.
- 🧠 Features: dos columnas, barras rojas (importancia global) y plata (SHAP de la carrera).
Sin datos/modelo → avisos (`st.info`/`model_missing`), nunca excepción.

- [ ] **Step 4: Run tests + verificación visual**

Run: `venv/bin/python -m pytest tests -v` → todo PASS.
Run: `venv/bin/streamlit run src/dashboard/app.py --server.headless true --server.port 8599` en background y captura con Firefox headless (`firefox --headless --screenshot`), revisar la imagen.

- [ ] **Step 5: Commit**

```bash
git add .streamlit/config.toml src/dashboard tests/test_dashboard.py
git commit -m "feat: rediseño del dashboard en rojo Fórmula y plata"
```

---

### Task 7: Datos 2024–2026, reentreno y cierre

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `AGENTS.md`, `reports/walk_forward_summary.json` (regenerado)

- [ ] **Step 1:** `venv/bin/python -m src.data.collect --years 2024 2025 2026` → parquets R/Q y `schedule.parquet` en las tres carpetas.
- [ ] **Step 2:** `make features`, `make train`, `make evaluate` con `venv/bin/python -m ...` → `reports/walk_forward_results.csv` con 2023, 2024, 2025.
- [ ] **Step 3:** `venv/bin/python -m src.models.upcoming` → muestra la próxima carrera.
- [ ] **Step 4:** `venv/bin/ruff check src tests`, `venv/bin/black --check src tests`, `venv/bin/python -m pytest tests -v` → todo verde.
- [ ] **Step 5:** README (Jolpica, `make update`, pestañas, próxima carrera), CLAUDE.md y AGENTS.md (arquitectura nueva).
- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md AGENTS.md reports/walk_forward_summary.json
git commit -m "docs: README y guías actualizadas; métricas walk-forward 2023-2025"
```
