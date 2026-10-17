"""Prepara los datos de OpenF1 de una carrera para la repetición animada.

Todo se pasa a segundos desde el inicio de la sesión. Las posiciones (x, y)
van a 1 Hz y el navegador interpola entre segundos; el resto (posición en
carrera, gaps, vueltas, estado de pista...) va como eventos [t, valor] que
el componente consulta con búsqueda binaria.
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.openf1 import to_timestamp

DEFAULT_COLOR = "#C0C0C0"
END_PADDING = 120  # segundos tras la bandera a cuadros (vuelta de honor)
INTERP_LIMIT = 5  # segundos sin datos que se rellenan interpolando
NO_LAP_END = 999  # stint todavía abierto


def _messages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(
            columns=["date", "category", "flag", "scope", "sector", "message"]
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    return df.sort_values("date").reset_index(drop=True)


def _is_vsc(message: str) -> bool:
    return "VSC" in message or "VIRTUAL" in message


def _message_kind(row) -> str:
    if row["category"] == "SafetyCar":
        return "VSC" if _is_vsc(str(row["message"]).upper()) else "SC"
    if isinstance(row.get("flag"), str) and row["flag"]:
        return row["flag"]
    return str(row["category"]).upper()


def track_status_timeline(race_control: pd.DataFrame) -> list[tuple]:
    """Estado de pista a partir de los mensajes de dirección de carrera.

    Devuelve [(timestamp, estado), ...] solo con los cambios; el primero lleva
    timestamp None (= inicio). Estados: GREEN, YELLOW, SC, VSC, RED, CHEQUERED.
    Las amarillas de sector se acumulan hasta que se despejan todas.
    """
    base, sectors = "GREEN", set()
    timeline: list[tuple] = [(None, "GREEN")]

    for _, row in _messages(race_control).iterrows():
        message = str(row["message"]).upper()
        flag, scope = row.get("flag"), row.get("scope")
        if row["category"] == "SafetyCar" and "DEPLOYED" in message:
            base = "VSC" if _is_vsc(message) else "SC"
        elif scope == "Track" and flag == "RED":
            base = "RED"
        elif scope == "Track" and flag in ("GREEN", "CLEAR"):
            base, sectors = "GREEN", set()
        elif scope == "Track" and flag == "CHEQUERED":
            base = "CHEQUERED"
        elif scope == "Sector" and flag in ("YELLOW", "DOUBLE YELLOW"):
            sectors.add(row.get("sector"))
        elif scope == "Sector" and flag == "CLEAR":
            sectors.discard(row.get("sector"))
        else:
            continue
        status = base if base != "GREEN" else ("YELLOW" if sectors else "GREEN")
        if status != timeline[-1][1]:
            timeline.append((row["date"], status))
    return timeline


def _seconds(ts, t0: pd.Timestamp) -> int:
    return max(0, round((ts - t0).total_seconds()))


def _events(df: pd.DataFrame, value_col: str, t0, convert=lambda v: v) -> list:
    """[[t, valor], ...] ordenado y sin repeticiones consecutivas."""
    if df.empty or value_col not in df.columns:
        return []
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    events: list = []
    for ts, value in df.sort_values("date")[["date", value_col]].itertuples(
        index=False
    ):
        value = convert(value)
        if value is None or (events and events[-1][1] == value):
            continue
        events.append([_seconds(ts, t0), value])
    return events


def _gap(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return str(value)


def _color(value) -> str:
    value = str(value or "")
    return (
        f"#{value.upper()}" if re.fullmatch(r"[0-9A-Fa-f]{6}", value) else DEFAULT_COLOR
    )


def _name(row) -> str:
    full = str(row.get("full_name") or row.get("name_acronym") or "")
    return full.title()


def _clean(values) -> list:
    return [None if pd.isna(v) else round(float(v)) for v in values]


def _for_driver(df: pd.DataFrame, number) -> pd.DataFrame:
    if df.empty or "driver_number" not in df.columns:
        return df.iloc[0:0]
    return df[df["driver_number"] == number]


def build_replay(session: dict, frames: dict, end_padding: int = END_PADDING) -> dict:
    """Payload JSON-serializable para el componente de repetición."""
    t0 = to_timestamp(session["date_start"])
    rc = _messages(frames["race_control"])
    chequered = rc[rc["flag"] == "CHEQUERED"]["date"]
    location = frames["location"].copy()
    if not location.empty:
        location["date"] = pd.to_datetime(location["date"], utc=True)
    if not chequered.empty:
        t_end = chequered.iloc[0] + pd.Timedelta(seconds=end_padding)
    elif not location.empty:
        t_end = location["date"].max()
    else:
        t_end = to_timestamp(session.get("date_end") or session["date_start"])
    duration = max(0, int((t_end - t0).total_seconds()))
    timeline = pd.date_range(t0, periods=duration + 1, freq="1s")

    laps = frames["laps"].copy()
    if not laps.empty:
        laps["date"] = laps["date_start"]

    drivers = []
    for _, d in frames["drivers"].iterrows():
        number = d["driver_number"]
        loc = _for_driver(location, number).set_index("date").sort_index()
        loc = loc[~loc.index.duplicated()]
        xy = (
            loc[["x", "y"]]
            .reindex(timeline)
            .interpolate(limit=INTERP_LIMIT, limit_area="inside")
        )
        stints = _for_driver(frames["stints"], number)
        pit = _for_driver(frames["pit"], number)
        pit_col = "lane_duration" if "lane_duration" in pit.columns else "pit_duration"
        drivers.append(
            {
                "num": str(number),
                "abbr": str(d.get("name_acronym") or number),
                "name": _name(d),
                "team": str(d.get("team_name") or ""),
                "color": _color(d.get("team_colour")),
                "x": _clean(xy["x"]),
                "y": _clean(xy["y"]),
                "pos": _events(
                    _for_driver(frames["position"], number), "position", t0, int
                ),
                "gap": _events(
                    _for_driver(frames["intervals"], number), "gap_to_leader", t0, _gap
                ),
                "laps": _events(_for_driver(laps, number), "lap_number", t0, int),
                "stints": [
                    [
                        int(s["lap_start"]),
                        NO_LAP_END if pd.isna(s["lap_end"]) else int(s["lap_end"]),
                        str(s["compound"] or "UNKNOWN"),
                    ]
                    for _, s in stints.iterrows()
                    if not pd.isna(s["lap_start"])
                ],
                "pits": [
                    [
                        _seconds(to_timestamp(p["date"]), t0),
                        float(p[pit_col]) if not pd.isna(p[pit_col]) else 20.0,
                    ]
                    for _, p in pit.iterrows()
                    if pit_col in pit.columns
                ],
            }
        )

    leader_laps = []
    if not laps.empty:
        firsts = laps.dropna(subset=["date_start"]).copy()
        firsts["date_start"] = pd.to_datetime(
            firsts["date_start"], utc=True, format="ISO8601"
        )
        for lap, ts in firsts.groupby("lap_number")["date_start"].min().items():
            leader_laps.append([_seconds(ts, t0), int(lap)])

    outline = frames["outline"]
    if outline.empty and not location.empty:
        first = location["driver_number"].iloc[0]
        outline = location[location["driver_number"] == first]

    return {
        "meta": {
            "name": str(session.get("circuit_short_name") or ""),
            "session_key": session.get("session_key"),
            "duration": duration,
            "total_laps": int(laps["lap_number"].max()) if not laps.empty else 0,
        },
        "outline": [
            [round(float(x)), round(float(y))] for x, y in outline[["x", "y"]].values
        ],
        "drivers": drivers,
        "leader_laps": sorted(leader_laps),
        "status": [
            [0 if ts is None else _seconds(ts, t0), status]
            for ts, status in track_status_timeline(rc)
        ],
        "messages": [
            [_seconds(row["date"], t0), str(row["message"]), _message_kind(row)]
            for _, row in rc.iterrows()
            if row["date"] >= t0
        ],
    }
