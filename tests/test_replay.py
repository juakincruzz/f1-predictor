"""Tests de la preparación de datos de la repetición."""

import pandas as pd

from src.data.replay import build_replay, track_status_timeline

T = "2026-10-11T12:{:02d}:{:02d}+00:00"


def _rc(rows):
    return pd.DataFrame(
        [
            {
                "date": T.format(m, s),
                "category": c,
                "flag": f,
                "scope": sc,
                "sector": sec,
                "message": msg,
            }
            for m, s, c, f, sc, sec, msg in rows
        ]
    )


def _statuses(df):
    return [status for _, status in track_status_timeline(df)]


def test_track_status_vsc_then_clear():
    df = _rc(
        [
            (3, 0, "SessionStatus", None, None, None, "SESSION STARTED"),
            (16, 0, "SafetyCar", None, None, None, "VSC DEPLOYED"),
            (17, 0, "SafetyCar", None, None, None, "VSC ENDING"),
            (17, 10, "Flag", "CLEAR", "Track", None, "TRACK CLEAR"),
        ]
    )
    assert _statuses(df) == ["GREEN", "VSC", "GREEN"]


def test_track_status_safety_car_red_and_chequered():
    df = _rc(
        [
            (5, 0, "SafetyCar", None, None, None, "SAFETY CAR DEPLOYED"),
            (9, 0, "Flag", "RED", "Track", None, "RED FLAG"),
            (20, 0, "Flag", "GREEN", "Track", None, "GREEN LIGHT - PIT EXIT OPEN"),
            (50, 0, "Flag", "CHEQUERED", "Track", None, "CHEQUERED FLAG"),
        ]
    )
    assert _statuses(df) == ["GREEN", "SC", "RED", "GREEN", "CHEQUERED"]


def test_track_status_sector_yellows_until_all_cleared():
    df = _rc(
        [
            (10, 0, "Flag", "YELLOW", "Sector", 6, "YELLOW IN TRACK SECTOR 6"),
            (10, 5, "Flag", "DOUBLE YELLOW", "Sector", 7, "DOUBLE YELLOW SECTOR 7"),
            (10, 30, "Flag", "CLEAR", "Sector", 6, "CLEAR IN TRACK SECTOR 6"),
            (10, 40, "Flag", "CLEAR", "Sector", 7, "CLEAR IN TRACK SECTOR 7"),
            (11, 0, "Flag", "BLUE", "Driver", None, "WAVED BLUE FLAG"),
        ]
    )
    assert _statuses(df) == ["GREEN", "YELLOW", "GREEN"]


def _frames():
    loc = pd.DataFrame(
        {
            "date": pd.to_datetime([T.format(0, s) for s in (0, 1, 3)] * 2, utc=True),
            "x": [0.0, 10.0, 30.0, 5.0, 6.0, 8.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "driver_number": [16, 16, 16, 1, 1, 1],
        }
    )
    return {
        "drivers": pd.DataFrame(
            {
                "driver_number": [16, 1],
                "name_acronym": ["LEC", "VER"],
                "full_name": ["Charles LECLERC", "Max VERSTAPPEN"],
                "team_name": ["Ferrari", "Red Bull Racing"],
                "team_colour": ["E8002D", None],
            }
        ),
        "location": loc,
        "outline": pd.DataFrame({"x": [0.0, 10.0, 20.0], "y": [0.0, 5.0, 0.0]}),
        "position": pd.DataFrame(
            {
                "date": [T.format(0, 0), T.format(0, 0), T.format(0, 2)],
                "driver_number": [16, 1, 1],
                "position": [1, 2, 1],
            }
        ),
        "intervals": pd.DataFrame(
            {
                "date": [T.format(0, 1), T.format(0, 2), T.format(0, 3)],
                "driver_number": [1, 1, 16],
                "gap_to_leader": [1.2345, 1.2345, "+1 LAP"],
            }
        ),
        "laps": pd.DataFrame(
            {
                "driver_number": [16, 1, 16],
                "lap_number": [1, 1, 2],
                "date_start": [T.format(0, 0), T.format(0, 1), T.format(0, 2)],
            }
        ),
        "stints": pd.DataFrame(
            {
                "driver_number": [16],
                "lap_start": [1],
                "lap_end": [2],
                "compound": ["MEDIUM"],
            }
        ),
        "pit": pd.DataFrame(
            {"date": [T.format(0, 2)], "driver_number": [1], "lane_duration": [21.5]}
        ),
        "race_control": _rc(
            [(0, 3, "Flag", "CHEQUERED", "Track", None, "CHEQUERED FLAG")]
        ),
    }


SESSION = {
    "session_key": 11388,
    "date_start": T.format(0, 0),
    "date_end": T.format(30, 0),
    "circuit_short_name": "Singapore",
}


def test_build_replay_payload():
    p = build_replay(SESSION, _frames(), end_padding=0)

    assert p["meta"]["duration"] == 3
    assert p["meta"]["total_laps"] == 2
    assert p["outline"] == [[0, 0], [10, 5], [20, 0]]
    lec, ver = p["drivers"]
    assert (lec["abbr"], lec["color"], ver["color"]) == ("LEC", "#E8002D", "#C0C0C0")
    assert lec["name"] == "Charles Leclerc"
    # El hueco del segundo 2 se interpola
    assert lec["x"] == [0, 10, 20, 30]
    assert lec["pos"] == [[0, 1]]
    assert ver["pos"] == [[0, 2], [2, 1]]
    # Solo cambios de gap; el texto se conserva
    assert ver["gap"] == [[1, 1.234]]
    assert lec["gap"] == [[3, "+1 LAP"]]
    assert lec["laps"] == [[0, 1], [2, 2]]
    assert lec["stints"] == [[1, 2, "MEDIUM"]]
    assert ver["pits"] == [[2, 21.5]]
    assert p["leader_laps"] == [[0, 1], [2, 2]]
    assert p["status"] == [[0, "GREEN"], [3, "CHEQUERED"]]
    assert p["messages"] == [[3, "CHEQUERED FLAG", "CHEQUERED"]]
