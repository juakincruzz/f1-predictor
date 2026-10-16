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
DRIVERS = [("1", "verstappen", "Red Bull Racing"), ("4", "norris", "McLaren")]


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
