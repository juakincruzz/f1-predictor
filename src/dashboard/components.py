"""Componentes HTML del dashboard (barra superior, celdas, podio, torre, rejilla).

Todo texto que viene de los datos pasa por `_e` antes de entrar en el HTML.
"""

import html
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

DEFAULT_TEAM_COLOR = "C0C0C0"
LABELS = {1: "P1", 2: "P2", 3: "P3"}
ROW_DELAY_MS, MAX_DELAY_MS = 35, 700

# Colores de sector: siempre el mismo significado en todo el dashboard
BEST, GOOD, WARN, BAD, OFF = (
    "var(--purple)",
    "var(--green)",
    "var(--yellow)",
    "var(--red)",
    "var(--dim)",
)
SECTOR_LEGEND = [
    (BEST, "Favorito / acierto exacto"),
    (GOOD, "En el podio"),
    (WARN, "Aviso o sorpresa"),
    (BAD, "Fallo"),
]


@dataclass
class Kpi:
    label: str
    value: str | int
    sub: str = ""
    tone: str = ""
    suffix: str = ""


def _e(value) -> str:
    return html.escape(str(value))


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _team_color(row: pd.Series) -> str:
    color = str(row.get("TeamColor", "") or "")
    return color if re.fullmatch(r"[0-9A-Fa-f]{6}", color) else DEFAULT_TEAM_COLOR


def _text(row: pd.Series, col: str, fallback: str = "") -> str:
    value = row.get(col)
    return fallback if value is None or pd.isna(value) else str(value)


def _prob(row: pd.Series, col: str) -> float:
    value = row.get(col, 0)
    return 0.0 if value is None or pd.isna(value) else float(value)


def _count(value: int, suffix: str = "") -> str:
    """Número que cuenta desde 0 al cargarse (CSS @property, sin JavaScript)."""
    return f'<span class="f1-num" style="--n:{int(value)}"></span>{_e(suffix)}'


def _delay(i: int) -> str:
    return f"animation-delay:{min(i * ROW_DELAY_MS, MAX_DELAY_MS)}ms"


def topbar(leds: list[tuple[str, str, bool]]) -> None:
    """Cabecera tipo muro de boxes con LEDs de estado (texto, color, parpadea)."""
    lights = "".join(
        f'<span class="f1-led{" live" if live else ""}" style="--c:{color}">'
        f"<i></i>{_e(text)}</span>"
        for text, color, live in leds
    )
    _html(
        '<div class="f1-top"><div class="f1-brand"><b>F1 Podium <span>Predictor</span>'
        f'</b><span>PIT WALL</span></div><div class="f1-leds">{lights}</div></div>'
    )


def context(label: str, title: str, details: str, tag: str, tone: str) -> None:
    """Línea «viendo»: qué carrera está seleccionada y en qué estado está."""
    _html(
        f'<div class="f1-context"><span>{_e(label)} ▸</span><b>{_e(title)}</b>'
        f'<span>{_e(details)}</span><span class="f1-tag" style="--c:{tone}">'
        f"{_e(tag)}</span></div>"
    )


def section(title: str, hint: str = "") -> None:
    hint_html = f'<div class="h">{_e(hint)}</div>' if hint else ""
    _html(f'<div class="f1-section"><div class="t">{_e(title)}</div>{hint_html}</div>')


def kpis(items: list[Kpi]) -> None:
    """Fila de celdas; los valores enteros cuentan desde 0 al aparecer."""
    cells = []
    for i, k in enumerate(items):
        numeric = isinstance(k.value, int)
        value = _count(k.value, k.suffix) if numeric else _e(k.value)
        tone = f"--tone:{k.tone};" if k.tone else ""
        sub = f'<div class="s">{_e(k.sub)}</div>' if k.sub else ""
        cells.append(
            f'<div class="f1-kpi" style="{tone}animation-delay:{i * 60}ms">'
            f'<div class="l">{_e(k.label)}</div>'
            f'<div class="v{"" if numeric else " text"}" title="{_e(k.value)}">{value}</div>'
            f"{sub}</div>"
        )
    _html(f'<div class="f1-kpis">{"".join(cells)}</div>')


def podium_cards(df: pd.DataFrame, prob_label: str = "Prob. podio") -> None:
    """Tres tarjetas P1/P2/P3 con las tres primeras filas del df, en orden.

    La barra inferior reparte la probabilidad de podio en P1 (morado),
    P2 (verde) y P3 (amarillo).
    """
    cards = []
    for pos, (_, row) in enumerate(df.head(3).iterrows(), 1):
        abbr = _text(row, "Abbreviation", "—")
        split = "".join(
            f'<i class="{cls}" style="width:{_prob(row, col) * 100:.1f}%"></i>'
            for cls, col in (("a", "Prob_P1"), ("b", "Prob_P2"), ("c", "Prob_P3"))
        )
        cards.append(
            f'<div class="f1-pcard p{pos}" style="--team:#{_team_color(row)}">'
            f'<div class="pos">P{pos}</div><div class="abbr">{_e(abbr)}</div>'
            f'<div class="name">{_e(_text(row, "FullName", abbr))}</div>'
            f'<div class="team">{_e(_text(row, "TeamName"))}</div>'
            f'<div class="prob"><span>{_e(prob_label)}</span>'
            f'<b>{_count(round(_prob(row, "PodiumProbability") * 100), "%")}</b></div>'
            f'<div class="f1-split" title="P1 / P2 / P3">{split}</div></div>'
        )
    _html(f'<div class="f1-podium">{"".join(cards)}</div>')


def _row_tone(i: int, prob: float) -> str:
    if i == 0:
        return BEST
    if i < 3:
        return GOOD
    return WARN if prob >= 0.25 else OFF


def _real_chip(i: int, target) -> str:
    """Chip con el resultado real comparado con lo que dijo el modelo."""
    in_top3 = i < 3
    if pd.isna(target) or int(target) == 0:
        return f'<span class="f1-chip" style="--c:{BAD}">✗</span>' if in_top3 else ""
    target = int(target)
    if in_top3:
        tone = BEST if target == i + 1 else GOOD
    else:
        tone = WARN
    return f'<span class="f1-chip" style="--c:{tone}">{LABELS[target]}</span>'


def timing_tower(preds: pd.DataFrame, with_actual: bool = False) -> None:
    """Torre de tiempos: un piloto por fila, ordenado por probabilidad de podio."""
    head = (
        '<div class="hd"><span>Pos</span><span></span><span>Pil.</span>'
        '<span class="n">Equipo</span><span class="m">Sal.</span>'
        '<span class="m">P1</span><span class="m">P2</span><span class="m">P3</span>'
        f'<span>Prob. podio</span>{"<span>Real</span>" if with_actual else ""}</div>'
    )
    rows = []
    for i, (_, row) in enumerate(preds.iterrows()):
        prob = _prob(row, "PodiumProbability")
        grid = row.get("grid_position")
        grid_txt = "—" if grid is None or pd.isna(grid) else f"P{int(grid)}"
        cells = "".join(
            f'<span class="m{"" if _prob(row, c) >= 0.1 else " dim"}">'
            f"{_prob(row, c):.0%}</span>"
            for c in ("Prob_P1", "Prob_P2", "Prob_P3")
        )
        real = f"<span>{_real_chip(i, row.get('target'))}</span>" if with_actual else ""
        rows.append(
            f'<div class="r{" top" if i == 0 else ""}" '
            f'style="--team:#{_team_color(row)};--tone:{_row_tone(i, prob)};{_delay(i)}">'
            f'<span class="p">{i + 1}</span><span class="c"></span>'
            f'<span class="a">{_e(_text(row, "Abbreviation", "—"))}</span>'
            f'<span class="n">{_e(_text(row, "TeamName"))}</span>'
            f'<span class="m dim">{grid_txt}</span>{cells}'
            f'<span class="bar"><span><i style="width:{prob * 100:.1f}%;{_delay(i)}">'
            f"</i></span><b>{prob:.0%}</b></span>{real}</div>"
        )
    real_col = "--real:52px" if with_actual else ""
    _html(f'<div class="f1-tower" style="{real_col}">{head}{"".join(rows)}</div>')


def legend(items: list[tuple[str, str]] = SECTOR_LEGEND) -> None:
    spans = "".join(f'<span style="--c:{c}">{_e(t)}</span>' for c, t in items)
    _html(f'<div class="f1-legend">{spans}</div>')


def season_grid(rounds: list[dict]) -> None:
    """Rejilla de rondas: código, nombre y aciertos de podio (●●○) o su estado."""
    cells = "".join(
        f'<div class="f1-round" style="--tone:{r["tone"]};{_delay(i)}">'
        f'<div class="k">R{r["round"]:02d} · {_e(r["code"])}</div>'
        f'<div class="e" title="{_e(r["name"])}">{_e(r["name"])}</div>'
        f'<div class="{"d" if r.get("dots") else "st"}">{_e(r["mark"])}</div></div>'
        for i, r in enumerate(rounds)
    )
    _html(f'<div class="f1-season">{cells}</div>')


def circuit_card(svg: str, left: str = "", right: str = "") -> None:
    """Tarjeta con el SVG del trazado (generado por circuit_map, sin texto externo)."""
    meta = (
        f'<div class="f1-circuit-meta"><span>{_e(left)}</span><span>{_e(right)}</span></div>'
        if left or right
        else ""
    )
    _html(f'<div class="f1-circuit-card">{svg}{meta}</div>')


def empty(title: str, body: str, tone: str = WARN, command: str = "") -> None:
    """Aviso de estado vacío con qué hacer para salir de él."""
    cmd = f" Ejecuta <code>{_e(command)}</code>." if command else ""
    _html(
        f'<div class="f1-empty" style="--c:{tone}"><b>{_e(title)}</b>'
        f"<p>{_e(body)}{cmd}</p></div>"
    )


def model_missing() -> None:
    empty(
        "No hay modelo entrenado",
        "Sin modelo no hay predicciones.",
        BAD,
        "make features && make train",
    )
