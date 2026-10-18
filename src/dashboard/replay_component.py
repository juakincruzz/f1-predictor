"""Componente HTML/JS de la repetición animada (canvas + torre de posiciones)."""

import json
from pathlib import Path

import streamlit as st

TEMPLATE = Path(__file__).parent / "assets" / "replay.html"
PLACEHOLDER = "__REPLAY_DATA__"


def replay_html(payload: dict) -> str:
    """HTML autónomo con el payload incrustado (sin poder cerrar el <script>)."""
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return TEMPLATE.read_text().replace(PLACEHOLDER, data.replace("</", "<\\/"))


def render_replay(payload: dict, height: int = 700) -> None:
    st.iframe(replay_html(payload), height=height)
