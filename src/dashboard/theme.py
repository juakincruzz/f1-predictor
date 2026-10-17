"""Estética F1 clásica para el dashboard: rojo Fórmula, gris plateado y carbono."""

import altair as alt
import streamlit as st

RED = "#E10600"
RED_DARK = "#A30500"
SILVER = "#C9CDD2"
SILVER_LIGHT = "#EEF0F2"
SILVER_DARK = "#8A8F96"
CARBON = "#0B0B0D"
PANEL = "#17181C"
LINE = "#2A2C31"
TEXT = "#D9D9DE"
FONT = "Titillium Web"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,700;1,900&display=swap');

html, body, .stApp, .stMarkdown, button, input, select, textarea,
[data-testid="stSidebar"], [data-baseweb="select"] {{
  font-family: '{FONT}', sans-serif !important;
}}
.stApp {{
  background:
    radial-gradient(ellipse at 15% -20%, #23242A 0%, rgba(11,11,13,0) 55%),
    repeating-linear-gradient(45deg, rgba(255,255,255,.012) 0 2px, transparent 2px 6px),
    {CARBON};
}}
[data-testid="stDecoration"] {{
  background: linear-gradient(90deg, {RED} 0%, {RED} 35%, {SILVER} 100%);
  height: 4px;
}}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #16171B 0%, #0E0F12 100%);
  border-right: 1px solid {LINE};
}}
[data-testid="stSidebar"] > div:first-child {{ border-top: 4px solid {RED}; }}
[data-testid="stSidebar"] a {{ color: {SILVER}; text-decoration-color: {RED}; }}
[data-testid="stSidebar"] a:hover {{ color: #fff; }}
[data-testid="stSidebar"] label p {{
  text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
  font-weight: 700; color: {SILVER_DARK};
}}

/* Cabecera: placa de metal cepillado con franjas rojas */
.f1-header {{
  position: relative; overflow: hidden; margin: 0 0 1.4rem;
  padding: 1.15rem 1.6rem 1rem; border-radius: 4px;
  background:
    repeating-linear-gradient(90deg, rgba(255,255,255,.10) 0 1px, rgba(0,0,0,.04) 1px 3px),
    linear-gradient(180deg, #F7F8F9 0%, #D5D8DC 38%, #A9AEB5 52%, #C9CDD2 72%, #EEF0F2 100%);
  border-bottom: 5px solid {RED};
  box-shadow: inset 0 1px 0 #fff, inset 0 -2px 0 rgba(0,0,0,.25), 0 10px 28px rgba(0,0,0,.55);
}}
.f1-header::after {{
  content: ""; position: absolute; top: 0; right: -30px; width: 280px; height: 100%;
  background: linear-gradient(105deg,
    transparent 0 20%, {RED} 20% 48%, transparent 48% 54%, {RED_DARK} 54% 66%, transparent 66%);
}}
.f1-title {{
  position: relative; z-index: 1; color: #141519; font-weight: 900; font-style: italic;
  text-transform: uppercase; letter-spacing: .01em; font-size: 2.3rem; line-height: 1;
}}
.f1-title span {{ color: {RED}; }}
.f1-header p {{
  position: relative; z-index: 1; margin: .4rem 0 0; color: #3A3D43; font-weight: 700;
  text-transform: uppercase; letter-spacing: .2em; font-size: .74rem;
}}

/* Títulos de sección */
.f1-section {{
  display: flex; align-items: center; gap: .7rem; margin: 1.5rem 0 .8rem;
  text-transform: uppercase; font-weight: 700; letter-spacing: .14em;
  color: {SILVER}; font-size: .86rem;
}}
.f1-section::before {{
  content: ""; width: 24px; height: 7px; background: {RED}; transform: skewX(-30deg);
}}
.f1-section::after {{
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, {SILVER_DARK}, transparent);
}}

/* Tarjetas de datos */
.f1-stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: .75rem; margin: .2rem 0 1rem;
}}
.f1-stat {{
  background: linear-gradient(180deg, #1C1D22, #141519);
  border: 1px solid {LINE}; border-top: 2px solid {SILVER_DARK};
  border-radius: 4px; padding: .7rem .95rem;
}}
.f1-stat.accent {{ border-top-color: {RED}; }}
.f1-stat .label {{
  font-size: .68rem; letter-spacing: .16em; text-transform: uppercase;
  color: {SILVER_DARK}; font-weight: 700;
}}
.f1-stat .value {{
  font-size: 1.3rem; font-weight: 700; color: {SILVER_LIGHT}; line-height: 1.25;
}}

/* Podio */
.f1-podium {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem;
  align-items: end; margin: .4rem 0 1.2rem;
}}
.f1-card {{
  position: relative; overflow: hidden; border-radius: 6px;
  padding: 1rem 1rem .95rem 1.25rem;
  background: linear-gradient(160deg, #222329 0%, #141519 65%);
  border: 1px solid #33353B;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.07), 0 12px 28px rgba(0,0,0,.5);
}}
.f1-card::before {{
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  background: var(--team);
}}
.f1-card.p1 {{ order: 2; min-height: 236px; border-color: {RED}; }}
.f1-card.p2 {{ order: 1; min-height: 206px; }}
.f1-card.p3 {{ order: 3; min-height: 184px; }}
.f1-pos {{
  display: inline-block; padding: .08rem .7rem; transform: skewX(-12deg);
  font-weight: 900; font-style: italic; font-size: 1.05rem; color: #141519;
  background: linear-gradient(180deg, {SILVER_LIGHT}, {SILVER_DARK});
}}
.p1 .f1-pos {{ background: {RED}; color: #fff; }}
.f1-abbr {{
  margin-top: .65rem; font-size: 2.7rem; font-weight: 900; font-style: italic;
  color: {SILVER_LIGHT}; line-height: 1;
}}
.f1-name {{ color: {TEXT}; font-weight: 600; margin-top: .15rem; }}
.f1-team {{
  color: {SILVER_DARK}; font-size: .8rem; text-transform: uppercase; letter-spacing: .1em;
}}
.f1-prob {{
  margin-top: .9rem; font-size: .68rem; color: {SILVER_DARK};
  text-transform: uppercase; letter-spacing: .14em; font-weight: 700;
}}
.f1-prob b {{ color: {SILVER_LIGHT}; font-size: 1.15rem; letter-spacing: 0; margin-left: .3rem; }}
.f1-bar {{ height: 6px; background: {LINE}; border-radius: 3px; overflow: hidden; margin-top: .35rem; }}
.f1-bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, {SILVER_DARK}, {SILVER_LIGHT}); }}
.p1 .f1-bar > span {{ background: linear-gradient(90deg, {RED_DARK}, {RED}); }}

/* Trazado del circuito */
.f1-circuit-card {{
  border: 1px solid {LINE}; border-radius: 6px; padding: 1rem;
  background: radial-gradient(ellipse at 50% 0%, #202127 0%, #111215 75%);
}}
.f1-circuit {{ display: block; width: 100%; max-height: 380px; }}

/* Pestañas */
[role="tablist"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
[data-testid="stTab"] {{ padding: .55rem 1rem; }}
[data-testid="stTab"] p {{
  text-transform: uppercase; letter-spacing: .1em; font-weight: 700;
  color: {SILVER_DARK}; font-size: .85rem;
}}
[data-testid="stTab"][aria-selected="true"] p {{ color: {SILVER_LIGHT}; }}
[data-testid="stTab"]:hover p {{ color: {SILVER_LIGHT}; }}
.react-aria-SelectionIndicator {{ background-color: {RED} !important; height: 3px !important; }}

/* Botones cromados */
.stButton > button {{
  border: 1px solid {SILVER_DARK}; border-radius: 3px; color: {SILVER_LIGHT};
  background: linear-gradient(180deg, #2C2E33, #1A1B1F);
  text-transform: uppercase; letter-spacing: .08em; font-weight: 700;
}}
.stButton > button:hover {{ border-color: {RED}; color: #fff; }}

@media (max-width: 640px) {{
  .f1-header::after {{ display: none; }}
  .f1-title {{ font-size: 1.7rem; }}
  .f1-podium {{ grid-template-columns: 1fr; }}
  .f1-card.p1, .f1-card.p2, .f1-card.p3 {{ min-height: 0; }}
  .f1-card.p1 {{ order: 1; }}
  .f1-card.p2 {{ order: 2; }}
  .f1-card.p3 {{ order: 3; }}
}}
"""


def apply_theme() -> None:
    """Inyecta el CSS del tema en la página."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Aplica la paleta del tema a un gráfico Altair."""
    return (
        chart.configure(background="transparent", font=FONT)
        .configure_axis(
            labelColor=SILVER_DARK,
            titleColor=SILVER,
            gridColor=LINE,
            domainColor=LINE,
            tickColor=LINE,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=600,
        )
        .configure_legend(
            labelColor=TEXT, titleColor=SILVER, orient="top", labelFontSize=12
        )
        .configure_view(strokeWidth=0)
    )
