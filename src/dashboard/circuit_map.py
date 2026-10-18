"""Trazado de circuitos como SVG a partir del GeoJSON de bacinger/f1-circuits (MIT)."""

import json
import math
from functools import lru_cache
from pathlib import Path

GEOJSON_PATH = Path(__file__).parent / "assets" / "f1-circuits.geojson"
VIEW_W, VIEW_H, PAD = 1000, 640, 40

# circuitId de Jolpica -> id del GeoJSON
CIRCUIT_IDS = {
    "albert_park": "au-1953",
    "americas": "us-2012",
    "bahrain": "bh-2002",
    "baku": "az-2016",
    "catalunya": "es-1991",
    "hockenheimring": "de-1932",
    "hungaroring": "hu-1986",
    "imola": "it-1953",
    "interlagos": "br-1940",
    "istanbul": "tr-2005",
    "jeddah": "sa-2021",
    "losail": "qa-2004",
    "madring": "es-2026",
    "marina_bay": "sg-2008",
    "miami": "us-2022",
    "monaco": "mc-1929",
    "monza": "it-1922",
    "mugello": "it-1914",
    "nurburgring": "de-1927",
    "portimao": "pt-2008",
    "red_bull_ring": "at-1969",
    "ricard": "fr-1969",
    "rodriguez": "mx-1962",
    "sepang": "my-1999",
    "shanghai": "cn-2004",
    "silverstone": "gb-1948",
    "sochi": "ru-2014",
    "spa": "be-1925",
    "suzuka": "jp-1962",
    "vegas": "us-2023",
    "villeneuve": "ca-1978",
    "yas_marina": "ae-2009",
    "zandvoort": "nl-1948",
}


# circuitId de Jolpica -> código corto para la tira de calendario
CIRCUIT_CODES = {
    "albert_park": "AUS",
    "americas": "USA",
    "bahrain": "BHR",
    "baku": "AZE",
    "buddh": "IND",
    "catalunya": "ESP",
    "hockenheimring": "GER",
    "hungaroring": "HUN",
    "imola": "EMI",
    "interlagos": "BRA",
    "istanbul": "TUR",
    "jeddah": "KSA",
    "losail": "QAT",
    "madring": "MAD",
    "marina_bay": "SGP",
    "miami": "MIA",
    "monaco": "MON",
    "monza": "ITA",
    "mugello": "TUS",
    "nurburgring": "EIF",
    "portimao": "POR",
    "red_bull_ring": "AUT",
    "ricard": "FRA",
    "rodriguez": "MEX",
    "sepang": "MAS",
    "shanghai": "CHN",
    "silverstone": "GBR",
    "sochi": "RUS",
    "spa": "BEL",
    "suzuka": "JPN",
    "valencia": "VAL",
    "vegas": "LVG",
    "villeneuve": "CAN",
    "yas_marina": "ABU",
    "yeongam": "KOR",
    "zandvoort": "NED",
}


def circuit_code(circuit_id: str, fallback: str = "") -> str:
    """Código de 3 letras del circuito; si no se conoce, las 3 primeras del fallback."""
    code = CIRCUIT_CODES.get(circuit_id or "")
    return code or (fallback or "???")[:3].upper()


@lru_cache(maxsize=1)
def load_geojson() -> dict:
    return json.loads(GEOJSON_PATH.read_text())


def _coordinates(circuit_id: str) -> list | None:
    geo_id = CIRCUIT_IDS.get(circuit_id)
    for feature in load_geojson()["features"]:
        if feature["properties"]["id"] == geo_id:
            return feature["geometry"]["coordinates"]
    return None


def circuit_svg(circuit_id: str) -> str | None:
    """SVG del trazado o None; la línea roja se dibuja animada desde la meta."""
    coords = _coordinates(circuit_id) if circuit_id else None
    if not coords:
        return None
    lat0 = math.radians(sum(lat for _, lat in coords) / len(coords))
    pts = [(lon * math.cos(lat0), -lat) for lon, lat in coords]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    dx, dy = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
    scale = min((VIEW_W - 2 * PAD) / dx, (VIEW_H - 2 * PAD) / dy)
    off_x = (VIEW_W - dx * scale) / 2
    off_y = (VIEW_H - dy * scale) / 2
    proj = [
        (off_x + (x - min(xs)) * scale, off_y + (y - min(ys)) * scale) for x, y in pts
    ]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in proj)
    sx, sy = proj[0]
    common = 'fill="none" stroke-linejoin="round" stroke-linecap="round"'
    return (
        f'<svg class="f1-circuit" viewBox="0 0 {VIEW_W} {VIEW_H}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Trazado del circuito">'
        f'<path d="{d}" {common} stroke="#1E2328" stroke-width="26"/>'
        f'<path d="{d}" {common} stroke="#59626B" stroke-width="9"/>'
        f'<path class="trace" pathLength="1" d="{d}" {common} stroke="#E10600" '
        'stroke-width="4"/>'
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="13" fill="#E10600" '
        'stroke="#F2F4F6" stroke-width="4"/>'
        "</svg>"
    )
