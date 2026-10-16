"""Componentes HTML del dashboard (cabecera, tarjetas, podio, tablas)."""

import html
import re

import pandas as pd
import streamlit as st

DEFAULT_TEAM_COLOR = "C0C0C0"
LABELS = {1: "P1", 2: "P2", 3: "P3"}
ROW_HEIGHT = 35  # altura de fila por defecto de st.dataframe


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


def header(main: str, accent: str, subtitle: str) -> None:
    _html(
        f'<div class="f1-header"><div class="f1-title">{_e(main)} '
        f"<span>{_e(accent)}</span></div><p>{_e(subtitle)}</p></div>"
    )


def section(title: str) -> None:
    _html(f'<div class="f1-section">{_e(title)}</div>')


def stat_cards(items: list[tuple[str, str]]) -> None:
    """Fila de tarjetas (etiqueta, valor); la primera va con acento rojo."""
    cards = "".join(
        f'<div class="f1-stat{" accent" if i == 0 else ""}">'
        f'<div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
        "</div>"
        for i, (label, value) in enumerate(items)
    )
    _html(f'<div class="f1-stats">{cards}</div>')


def podium_cards(df: pd.DataFrame, prob_label: str = "Prob. podio") -> None:
    """Tres tarjetas P1/P2/P3 con las tres primeras filas del df, en orden."""
    cards = []
    for pos, (_, row) in enumerate(df.head(3).iterrows(), 1):
        abbr = _text(row, "Abbreviation", "—")
        prob = float(row.get("PodiumProbability", 0) or 0)
        cards.append(
            f'<div class="f1-card p{pos}" style="--team:#{_team_color(row)}">'
            f'<span class="f1-pos">P{pos}</span>'
            f'<div class="f1-abbr">{_e(abbr)}</div>'
            f'<div class="f1-name">{_e(_text(row, "FullName", abbr))}</div>'
            f'<div class="f1-team">{_e(_text(row, "TeamName"))}</div>'
            f'<div class="f1-prob">{_e(prob_label)}<b>{prob:.0%}</b></div>'
            f'<div class="f1-bar"><span style="width:{prob * 100:.1f}%"></span></div>'
            "</div>"
        )
    _html(f'<div class="f1-podium">{"".join(cards)}</div>')


def prediction_table(preds: pd.DataFrame, with_actual: bool = False) -> None:
    """Tabla de probabilidades por piloto, ordenada como llega."""
    table = pd.DataFrame(
        {
            "Pos": range(1, len(preds) + 1),
            "Piloto": preds["Abbreviation"].to_numpy(),
            "Equipo": preds["TeamName"].to_numpy(),
            "Salida": preds["grid_position"].to_numpy(),
            "P1": preds["Prob_P1"].to_numpy() * 100,
            "P2": preds["Prob_P2"].to_numpy() * 100,
            "P3": preds["Prob_P3"].to_numpy() * 100,
            "Podio": preds["PodiumProbability"].to_numpy() * 100,
        }
    )
    if with_actual:
        table["Real"] = preds["target"].map(LABELS).fillna("—").to_numpy()
    pct = "%.1f%%"
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=ROW_HEIGHT * (len(table) + 1) + 3,
        column_config={
            "Pos": st.column_config.NumberColumn(width="small"),
            "Salida": st.column_config.NumberColumn(format="%d", width="small"),
            "P1": st.column_config.NumberColumn(format=pct),
            "P2": st.column_config.NumberColumn(format=pct),
            "P3": st.column_config.NumberColumn(format=pct),
            "Podio": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100
            ),
        },
    )


def model_missing() -> None:
    st.error("No hay modelo entrenado. Ejecuta `make features` y `make train`.")
