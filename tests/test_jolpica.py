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
        "Circuit": {
            "circuitId": f"c{rnd}",
            "Location": {"country": "X", "locality": "Y"},
        },
        key: results,
    }


def _page(total, races):
    return {"MRData": {"total": str(total), "RaceTable": {"Races": races}}}


def test_fetch_races_merges_race_split_across_pages(monkeypatch):
    ver = _entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)
    nor = _entry("4", "NOR", "norris", "mclaren", "McLaren", 2)
    pages = [_page(3, [_race(1, [ver])]), _page(3, [_race(1, [nor]), _race(2, [ver])])]
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
    races = [
        _race(
            3,
            [
                _entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1, "2"),
                _entry("30", None, "lawson", "rb", "RB F1 Team", 2, grid="0"),
            ],
        )
    ]
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
    ver = _entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)
    races = [_race(1, [ver], key="QualifyingResults")]
    df = jolpica.parse_session(races, 2026, "Q")
    assert df["Position"].tolist() == [1.0]
    assert "GridPosition" not in df.columns


def test_parse_schedule():
    sched = jolpica.parse_schedule([_race(1, []), _race(2, [])], 2026)
    assert sched["RoundNumber"].tolist() == [1, 2]
    assert sched["EventName"].tolist() == ["GP 1", "GP 2"]
    assert sched["Year"].unique().tolist() == [2026]
    assert sched["CircuitId"].tolist() == ["c1", "c2"]
    assert pd.api.types.is_datetime64_any_dtype(sched["EventDate"])
    # sin hora en la respuesta: cadena vacía
    assert sched["EventTime"].tolist() == ["", ""]


def test_parse_schedule_keeps_start_time():
    race = {**_race(1, []), "time": "05:00:00Z"}
    assert jolpica.parse_schedule([race], 2026)["EventTime"].tolist() == ["05:00:00Z"]


def test_collect_year_jolpica_writes_and_is_incremental(monkeypatch, tmp_path):
    ver = _entry("1", "VER", "max_verstappen", "red_bull", "Red Bull", 1)

    def fake_fetch(year, endpoint=""):
        if endpoint == "":
            return [_race(1, []), _race(2, [])]
        key = "Results" if endpoint == "results" else "QualifyingResults"
        # Solo la ronda 1 se ha disputado
        return [_race(1, [ver], key)]

    monkeypatch.setattr(jolpica, "fetch_races", fake_fetch)
    monkeypatch.setattr(jolpica, "get_raw_path", lambda sub=None: tmp_path)

    stats = jolpica.collect_year_jolpica(2026)
    assert (tmp_path / "schedule.parquet").exists()
    assert (tmp_path / "R01_R.parquet").exists()
    assert (tmp_path / "R01_Q.parquet").exists()
    assert not (tmp_path / "R02_R.parquet").exists()
    assert stats == {"processed": 2, "skipped": 0, "pending": 2, "total": 4}
    assert pd.read_parquet(tmp_path / "R01_R.parquet")["TotalRounds"].tolist() == [2]

    stats = jolpica.collect_year_jolpica(2026)
    assert stats == {"processed": 0, "skipped": 2, "pending": 2, "total": 4}
