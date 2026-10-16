"""Smoke test del dashboard: debe renderizar sin excepciones con o sin datos."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    labels = [tab.label for tab in at.tabs]
    assert labels == [
        "🏁 Próxima carrera",
        "🔮 Predicción",
        "📊 Histórico",
        "🧠 Features",
    ]
