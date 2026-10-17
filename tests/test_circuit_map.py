"""Tests del trazado de circuitos (GeoJSON) y del HTML de la repetición."""

from src.dashboard.circuit_map import CIRCUIT_IDS, circuit_svg, load_geojson
from src.dashboard.replay_component import replay_html

CIRCUITS_2026 = [
    "albert_park",
    "americas",
    "baku",
    "catalunya",
    "hungaroring",
    "interlagos",
    "jeddah",
    "losail",
    "madring",
    "marina_bay",
    "miami",
    "monaco",
    "monza",
    "red_bull_ring",
    "rodriguez",
    "sepang",
    "shanghai",
    "silverstone",
    "spa",
    "suzuka",
    "vegas",
    "villeneuve",
    "yas_marina",
    "zandvoort",
]


def test_every_2026_circuit_has_a_layout():
    geo_ids = {f["properties"]["id"] for f in load_geojson()["features"]}
    missing = [c for c in CIRCUITS_2026 if CIRCUIT_IDS.get(c) not in geo_ids]
    assert missing == []


def test_circuit_svg_draws_path_and_start_marker():
    svg = circuit_svg("marina_bay")
    assert svg.startswith("<svg")
    assert "<path" in svg and "<circle" in svg


def test_circuit_svg_unknown_circuit_is_none():
    assert circuit_svg("nurburgring_nordschleife") is None
    assert circuit_svg("") is None


def test_replay_html_embeds_payload_safely():
    html = replay_html({"meta": {"name": "</script><b>x"}, "drivers": []})
    assert "__REPLAY_DATA__" not in html
    assert "</script><b>" not in html
    assert "<\\/script><b>x" in html
