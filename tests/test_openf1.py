"""Tests del cliente OpenF1 (sin red: se parchea requests / get_json)."""

from datetime import date

import pandas as pd

from src.data import openf1

SESSIONS = [
    {"session_key": 11388, "date_start": "2026-10-11T12:00:00+00:00"},
    {"session_key": 11420, "date_start": "2026-11-22T04:00:00+00:00"},
]


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("no debería llamarse")


def test_get_json_treats_404_as_empty(monkeypatch):
    monkeypatch.setattr(openf1, "MIN_INTERVAL", 0)
    monkeypatch.setattr(
        openf1.requests, "get", lambda *a, **k: _Resp(404, {"detail": "No results"})
    )
    assert openf1.get_json("pit", {"session_key": 1}) == []


def test_get_json_retries_on_429(monkeypatch):
    responses = [_Resp(429, {}), _Resp(200, [{"ok": 1}])]
    monkeypatch.setattr(openf1, "MIN_INTERVAL", 0)
    monkeypatch.setattr(openf1, "RETRY_BASE", 0)
    monkeypatch.setattr(openf1.requests, "get", lambda *a, **k: responses.pop(0))
    assert openf1.get_json("pit", {}) == [{"ok": 1}]


def test_match_session_by_date_with_one_day_tolerance():
    assert openf1.match_session(SESSIONS, date(2026, 10, 11))["session_key"] == 11388
    # Las Vegas: sábado local, domingo en UTC
    assert openf1.match_session(SESSIONS, date(2026, 11, 21))["session_key"] == 11420
    assert openf1.match_session(SESSIONS, date(2026, 10, 1)) is None


def test_downsample_location_drops_zeros_and_resamples_to_1s():
    rows = [
        {"date": "2026-10-11T12:00:00.100+00:00", "x": 0, "y": 0},
        {"date": "2026-10-11T12:00:00.300+00:00", "x": 10, "y": 20},
        {"date": "2026-10-11T12:00:00.700+00:00", "x": 30, "y": 40},
        {"date": "2026-10-11T12:00:01.200+00:00", "x": 50, "y": 60},
    ]
    df = openf1.downsample_location(rows)
    assert df["x"].tolist() == [20.0, 50.0]
    assert df["y"].tolist() == [30.0, 60.0]
    assert str(df["date"].iloc[0]) == "2026-10-11 12:00:00+00:00"


def test_fastest_lap_window():
    laps = pd.DataFrame(
        {
            "driver_number": [16, 16, 16],
            "lap_number": [1, 2, 3],
            "date_start": [
                "2026-10-11T12:03:00+00:00",
                "2026-10-11T12:05:00+00:00",
                "2026-10-11T12:06:40+00:00",
            ],
            "lap_duration": [None, 100.0, 101.0],
        }
    )
    start, end = openf1.fastest_lap_window(laps, 16)
    assert str(start) == "2026-10-11 12:05:00+00:00"
    assert str(end) == "2026-10-11 12:06:40+00:00"


def test_download_race_writes_cache_and_is_reusable(monkeypatch, tmp_path):
    calls = []
    data = {
        "sessions": SESSIONS,
        "drivers": [{"driver_number": 16, "name_acronym": "LEC"}],
        "laps": [
            {
                "driver_number": 16,
                "lap_number": 1,
                "date_start": "2026-10-11T12:03:00+00:00",
                "lap_duration": 90.0,
            }
        ],
        "location": [
            {"date": "2026-10-11T12:03:00.5+00:00", "x": 1, "y": 2},
            {"date": "2026-10-11T12:04:00.5+00:00", "x": 3, "y": 4},
        ],
    }

    def fake_get(endpoint, params):
        calls.append(endpoint)
        return data.get(endpoint, [])

    monkeypatch.setattr(openf1, "get_json", fake_get)
    monkeypatch.setattr(openf1, "OPENF1_ROOT", tmp_path)

    session = openf1.download_race(2026, 17, date(2026, 10, 11))
    assert session["session_key"] == 11388
    assert openf1.is_downloaded(2026, 17)

    loaded_session, frames = openf1.load_race(2026, 17)
    assert loaded_session["session_key"] == 11388
    assert set(frames["location"]["driver_number"]) == {16}
    assert len(frames["outline"]) == 2
    assert {"drivers", "position", "intervals", "stints", "pit"} <= set(frames)

    n_calls = len(calls)
    openf1.download_race(2026, 17, date(2026, 10, 11))
    assert len(calls) == n_calls  # todo en caché: ninguna petición nueva
