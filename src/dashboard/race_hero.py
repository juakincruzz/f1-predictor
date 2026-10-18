"""Cabecera animada de la próxima carrera: semáforo de salida y cuenta atrás."""

import json
from pathlib import Path

import streamlit as st

TEMPLATE = Path(__file__).parent / "assets" / "hero.html"
PLACEHOLDER = "__HERO_DATA__"


def hero_html(payload: dict) -> str:
    """HTML autónomo con el payload incrustado (sin poder cerrar el <script>)."""
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return TEMPLATE.read_text().replace(PLACEHOLDER, data.replace("</", "<\\/"))


def render_hero(payload: dict) -> None:
    st.iframe(hero_html(payload), height="content")
