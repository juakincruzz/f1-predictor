"""Smoke test del dashboard: debe renderizar sin excepciones con o sin datos."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
SECTIONS = ["Carrera", "Temporada", "Modelo"]


def test_dashboard_renders_every_section_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    nav = at.segmented_control(key="_nav")
    assert nav.options == SECTIONS
    assert nav.value == "Carrera"
    if at.tabs:  # sin datos descargados solo hay un aviso
        assert [tab.label for tab in at.tabs] == ["Predicción", "Repetición", "Por qué"]

    for section in SECTIONS[1:] + SECTIONS[:1]:
        at.segmented_control(key="_nav").set_value(section).run()
        assert not at.exception
        assert at.query_params["seccion"] == [section]


def test_dashboard_switches_race_from_calendar_strip():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    strip = [p for p in at.pills if p.key == "_race"]
    if not strip or len(strip[0].options) < 2:
        return  # sin calendario no hay nada que cambiar
    # las opciones se muestran como «01 AUS»; el valor es el número de ronda
    target = int(strip[0].options[0].split()[0])
    strip[0].set_value(target).run()
    assert not at.exception
    assert at.session_state["round"] == target
