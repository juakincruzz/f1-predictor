"""Estética «muro de boxes»: pantalla de ingeniero con colores de sector F1.

Los colores de sector tienen siempre el mismo significado en todo el dashboard:
morado = lo mejor (favorito, acierto exacto), verde = bueno (en el podio),
amarillo = aviso (dato estimado, sorpresa), rojo = marca F1 y fallos.
"""

import altair as alt
import streamlit as st

RED = "#E10600"
PURPLE = "#B36BFF"
GREEN = "#36E07A"
YELLOW = "#F5D90A"
SILVER = "#C9CDD2"
BG = "#07080A"
PANEL = "#0D0F12"
PANEL_HI = "#13161A"
LINE = "#1E2328"
TEXT = "#D5DADF"
MUTED = "#7D868F"
DIM = "#4A525A"
WHITE = "#F2F4F6"
FONT = "Titillium Web"
MONO = "JetBrains Mono"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --red: {RED}; --purple: {PURPLE}; --green: {GREEN}; --yellow: {YELLOW};
  --bg: {BG}; --panel: {PANEL}; --panel-hi: {PANEL_HI}; --line: {LINE};
  --text: {TEXT}; --muted: {MUTED}; --dim: {DIM}; --white: {WHITE};
  --mono: '{MONO}', ui-monospace, monospace;
  --ease: cubic-bezier(.2, .8, .2, 1);
}}
html, body, .stApp, .stMarkdown, button, input, select, textarea,
[data-testid="stSidebar"], [data-baseweb="select"] {{
  font-family: '{FONT}', sans-serif !important;
}}
.stApp {{
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,.018) 0 1px, transparent 1px 3px),
    radial-gradient(ellipse at 50% -10%, #12161B 0%, rgba(7,8,10,0) 60%),
    {BG};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{
  background: linear-gradient(90deg, {RED} 0 22%, {PURPLE} 22% 44%, {GREEN} 44% 66%,
    {YELLOW} 66% 88%, {RED} 88%);
  height: 3px;
}}
.block-container {{ padding-top: 2.6rem; max-width: 1320px; }}
@keyframes f1-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes f1-slide {{ from {{ opacity: 0; transform: translateX(-10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes f1-fill {{ from {{ transform: scaleX(0); }} }}
@keyframes f1-blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .35; }} }}
@keyframes f1-sweep {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(100%); }} }}
@property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes f1-count {{ from {{ --n: 0; }} }}

/* Barra superior */
.f1-top {{
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  flex-wrap: wrap; padding: .55rem .9rem; margin-bottom: .9rem;
  border: 1px solid var(--line); border-left: 4px solid var(--red); border-radius: 4px;
  background: linear-gradient(90deg, var(--panel-hi), var(--panel));
  font-family: var(--mono); font-size: .78rem; color: var(--muted);
}}
.f1-brand {{ display: flex; align-items: baseline; gap: .7rem; }}
.f1-brand b {{
  font-family: '{FONT}', sans-serif; font-weight: 900; font-style: italic; font-size: 1.35rem;
  text-transform: uppercase; color: var(--white); letter-spacing: .01em;
}}
.f1-brand b span {{ color: var(--red); }}
.f1-leds {{ display: flex; gap: 1.1rem; flex-wrap: wrap; }}
.f1-led {{ display: inline-flex; align-items: center; gap: .4rem; text-transform: uppercase; letter-spacing: .08em; }}
.f1-led i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c); }}
.f1-led.live i {{ animation: f1-blink 1.6s ease-in-out infinite; }}

/* Navegación por secciones (segmented control) */
.st-key-_nav [data-testid="stButtonGroup"] {{ gap: 0; }}
.st-key-_nav button {{
  border-radius: 0 !important; border: 1px solid var(--line) !important; margin-left: -1px;
  background: var(--panel) !important; padding: .35rem 1.15rem !important;
  transition: background .2s, color .2s;
}}
.st-key-_nav button p {{
  font-family: var(--mono); text-transform: uppercase; letter-spacing: .12em;
  font-size: .78rem; color: var(--muted); font-weight: 700;
}}
.st-key-_nav button:hover p {{ color: var(--white); }}
.st-key-_nav button[aria-checked="true"] {{
  background: var(--panel-hi) !important; box-shadow: inset 0 -2px 0 var(--red);
}}
.st-key-_nav button[aria-checked="true"] p {{ color: var(--white); }}

/* Tira de calendario (pills) */
.st-key-_race [data-testid="stButtonGroup"] > div {{ gap: 4px; flex-wrap: wrap; }}
.st-key-_race button {{
  min-width: 54px; border-radius: 3px !important; padding: .2rem .45rem !important;
  border: 1px solid var(--line) !important; background: var(--panel) !important;
  box-shadow: inset 0 -3px 0 var(--tone, var(--dim));
  transition: transform .15s var(--ease), border-color .15s, background .15s;
}}
.st-key-_race button:hover {{ transform: translateY(-2px); border-color: var(--muted) !important; }}
.st-key-_race button p {{
  font-family: var(--mono); font-size: .72rem; line-height: 1.15; color: var(--text);
  white-space: pre; text-align: center;
}}
.st-key-_race button[aria-checked="true"] {{
  background: var(--panel-hi) !important; border-color: var(--white) !important;
  box-shadow: inset 0 -3px 0 var(--tone, var(--dim)), 0 0 0 1px var(--white);
}}
.st-key-_race button[aria-checked="true"] p {{ color: var(--white); font-weight: 700; }}

/* Etiquetas de widgets */
[data-testid="stWidgetLabel"] p {{
  font-family: var(--mono); text-transform: uppercase; letter-spacing: .12em;
  font-size: .68rem !important; color: var(--muted);
}}

/* Contexto «viendo» */
.f1-context {{
  position: relative; overflow: hidden; display: flex; align-items: center; gap: .8rem;
  flex-wrap: wrap; margin: .2rem 0 1rem; padding: .55rem .9rem;
  border: 1px solid var(--line); border-radius: 4px; background: var(--panel);
  font-family: var(--mono); font-size: .8rem; color: var(--muted);
}}
.f1-context::after {{
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.07), transparent);
  animation: f1-sweep .8s var(--ease) both;
}}
.f1-context b {{ color: var(--white); font-family: '{FONT}', sans-serif; font-size: 1.05rem; }}
.f1-tag {{
  display: inline-block; padding: .05rem .5rem; border-radius: 2px; font-size: .68rem;
  font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--c); border: 1px solid var(--c);
}}

/* Títulos de sección */
.f1-section {{ margin: 1.4rem 0 .7rem; animation: f1-in .4s var(--ease) both; }}
.f1-section .t {{
  display: flex; align-items: center; gap: .6rem; font-family: var(--mono);
  text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; color: var(--text);
  font-weight: 700;
}}
.f1-section .t::before {{ content: "//"; color: var(--red); }}
.f1-section .t::after {{ content: ""; flex: 1; height: 1px; background: var(--line); }}
.f1-section .h {{ margin-top: .2rem; color: var(--muted); font-size: .85rem; }}

/* Celdas de datos */
.f1-kpis {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 4px;
  overflow: hidden; margin: .2rem 0 1rem;
}}
.f1-kpi {{ background: var(--panel); padding: .7rem .9rem; animation: f1-in .45s var(--ease) both; }}
.f1-kpi .l {{
  font-family: var(--mono); text-transform: uppercase; letter-spacing: .12em;
  font-size: .66rem; color: var(--muted);
}}
.f1-kpi .v {{
  font-family: var(--mono); font-weight: 700; font-size: 1.5rem; color: var(--tone, var(--white));
  line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.f1-kpi .v.text {{ font-family: '{FONT}', sans-serif; font-size: 1.3rem; }}
.f1-kpi .s {{ font-size: .78rem; color: var(--muted); }}
.f1-num {{ animation: f1-count 1.1s var(--ease) both; counter-reset: n var(--n); }}
.f1-num::before {{ content: counter(n); }}

/* Podio */
.f1-podium {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; align-items: end;
  margin-bottom: 1rem;
}}
.f1-pcard {{
  position: relative; overflow: hidden; padding: .8rem .9rem .85rem;
  border: 1px solid var(--line); border-top: 3px solid var(--team); border-radius: 4px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--team) 14%, var(--panel-hi)), var(--panel) 70%);
  animation: f1-in .6s var(--ease) both; transition: transform .2s var(--ease), border-color .2s;
}}
.f1-pcard:hover {{ transform: translateY(-3px); border-color: var(--muted); }}
.f1-pcard.p1 {{ order: 2; min-height: 214px; animation-delay: .05s; }}
.f1-pcard.p2 {{ order: 1; min-height: 190px; animation-delay: .2s; }}
.f1-pcard.p3 {{ order: 3; min-height: 172px; animation-delay: .35s; }}
.f1-pcard .pos {{
  font-family: var(--mono); font-weight: 700; font-size: .8rem; letter-spacing: .1em;
  color: var(--muted);
}}
.f1-pcard.p1 .pos {{ color: var(--purple); }}
.f1-pcard .abbr {{
  font-weight: 900; font-style: italic; font-size: 2.6rem; color: var(--white); line-height: 1;
  margin-top: .3rem;
}}
.f1-pcard .name {{ font-weight: 600; color: var(--text); }}
.f1-pcard .team {{ font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
.f1-pcard .prob {{ margin-top: .7rem; display: flex; justify-content: space-between; align-items: baseline; }}
.f1-pcard .prob > span {{ font-family: var(--mono); font-size: .66rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }}
.f1-pcard .prob b {{ font-family: var(--mono); font-size: 1.3rem; color: var(--white); }}
.f1-pcard.p1 .prob b {{ color: var(--purple); }}
.f1-split {{ display: flex; height: 6px; margin-top: .35rem; border-radius: 2px; overflow: hidden; background: var(--line); }}
.f1-split i {{ display: block; height: 100%; transform-origin: left; animation: f1-fill 1s .3s var(--ease) both; }}
.f1-split .a {{ background: var(--purple); }}
.f1-split .b {{ background: var(--green); }}
.f1-split .c {{ background: var(--yellow); }}

/* Torre de tiempos */
.f1-tower {{ border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: var(--panel); }}
.f1-tower .hd, .f1-tower .r {{
  display: grid; align-items: center; gap: .6rem; padding: .32rem .8rem;
  grid-template-columns: 26px 4px 48px minmax(0, 1fr) 40px repeat(3, 46px) minmax(90px, 1.2fr) var(--real, 0);
}}
.f1-tower .hd {{
  font-family: var(--mono); font-size: .64rem; text-transform: uppercase; letter-spacing: .1em;
  color: var(--muted); background: var(--panel-hi); border-bottom: 1px solid var(--line);
}}
.f1-tower .r {{
  border-bottom: 1px solid rgba(30,35,40,.7); font-size: .88rem;
  animation: f1-slide .4s var(--ease) both; transition: background .15s;
}}
.f1-tower .r:hover {{ background: var(--panel-hi); }}
.f1-tower .r:last-child {{ border-bottom: 0; }}
.f1-tower .r.top {{ background: linear-gradient(90deg, rgba(179,107,255,.08), transparent 60%); }}
.f1-tower .p {{ font-family: var(--mono); font-weight: 700; color: var(--white); text-align: right; }}
.f1-tower .c {{ height: 18px; border-radius: 1px; background: var(--team); }}
.f1-tower .a {{ font-weight: 700; color: var(--white); letter-spacing: .04em; }}
.f1-tower .n {{ color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: .82rem; }}
.f1-tower .m {{ font-family: var(--mono); font-size: .78rem; color: var(--text); text-align: right; }}
.f1-tower .m.dim {{ color: var(--muted); }}
.f1-tower .bar {{ display: flex; align-items: center; gap: .5rem; }}
.f1-tower .bar > span {{ flex: 1; height: 6px; border-radius: 2px; background: var(--line); overflow: hidden; }}
.f1-tower .bar > span > i {{
  display: block; height: 100%; background: var(--tone); transform-origin: left;
  animation: f1-fill .9s var(--ease) both; animation-delay: inherit;
}}
.f1-tower .bar b {{ font-family: var(--mono); font-size: .82rem; color: var(--tone); min-width: 3.2em; text-align: right; }}
.f1-chip {{
  font-family: var(--mono); font-size: .66rem; font-weight: 700; letter-spacing: .06em;
  padding: .05rem .35rem; border-radius: 2px; text-align: center; white-space: nowrap;
  color: var(--bg); background: var(--c);
}}
.f1-legend {{
  display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; margin: .5rem 0 0; font-size: .78rem; color: var(--muted);
}}
.f1-legend span::before {{
  content: ""; display: inline-block; width: 9px; height: 9px; margin-right: .4rem;
  border-radius: 2px; background: var(--c); vertical-align: middle;
}}

/* Rejilla de temporada */
.f1-season {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 6px; }}
.f1-round {{
  padding: .55rem .65rem; border: 1px solid var(--line); border-left: 3px solid var(--tone);
  border-radius: 3px; background: var(--panel); animation: f1-in .4s var(--ease) both;
  transition: transform .15s var(--ease), background .15s;
}}
.f1-round:hover {{ transform: translateY(-2px); background: var(--panel-hi); }}
.f1-round .k {{ font-family: var(--mono); font-size: .68rem; color: var(--muted); }}
.f1-round .e {{ font-weight: 700; color: var(--white); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.f1-round .d {{ font-family: var(--mono); letter-spacing: .15em; color: var(--tone); }}
.f1-round .st {{ font-family: var(--mono); font-size: .72rem; color: var(--tone); text-transform: uppercase; }}

/* Circuito */
.f1-circuit-card {{
  border: 1px solid var(--line); border-radius: 4px; padding: .8rem; background:
    radial-gradient(ellipse at 50% 0%, #151A20 0%, var(--panel) 75%);
}}
.f1-circuit {{ display: block; width: 100%; max-height: 330px; }}
.f1-circuit .trace {{ stroke-dasharray: 1; stroke-dashoffset: 1; animation: f1-draw 2.4s .2s var(--ease) forwards; }}
@keyframes f1-draw {{ to {{ stroke-dashoffset: 0; }} }}
.f1-circuit-meta {{ display: flex; justify-content: space-between; margin-top: .4rem; font-family: var(--mono); font-size: .72rem; color: var(--muted); text-transform: uppercase; }}

/* Estado vacío */
.f1-empty {{
  padding: 1.2rem 1.3rem; border: 1px dashed var(--line); border-left: 3px solid var(--c, var(--yellow));
  border-radius: 4px; background: var(--panel); animation: f1-in .4s var(--ease) both;
}}
.f1-empty b {{ display: block; color: var(--white); font-size: 1.05rem; margin-bottom: .2rem; }}
.f1-empty p {{ margin: 0; color: var(--muted); }}
.f1-empty code {{ font-family: var(--mono); color: var(--text); background: var(--panel-hi); padding: .05rem .35rem; border-radius: 2px; }}

/* Pestañas */
[role="tablist"] {{ gap: .2rem; border-bottom: 1px solid var(--line); }}
[data-testid="stTab"] {{ padding: .5rem .95rem; }}
[data-testid="stTab"] p {{
  font-family: var(--mono); text-transform: uppercase; letter-spacing: .1em; font-weight: 700;
  color: var(--muted); font-size: .78rem;
}}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"]:hover p {{ color: var(--white); }}
.react-aria-SelectionIndicator {{ background-color: {RED} !important; height: 2px !important; }}

/* Botones */
.stButton > button {{
  border: 1px solid var(--line); border-radius: 3px; background: var(--panel-hi);
  transition: border-color .15s, transform .15s var(--ease);
}}
.stButton > button p {{ font-family: var(--mono); text-transform: uppercase; letter-spacing: .08em; font-size: .76rem; font-weight: 700; }}
.stButton > button:hover {{ border-color: var(--red); color: var(--white); }}
.stButton > button:active {{ transform: scale(.97); }}
[data-testid="stCaptionContainer"] {{ color: var(--muted); }}
[data-testid="stCaptionContainer"] p {{ font-size: .82rem; }}

@media (max-width: 760px) {{
  .f1-podium {{ grid-template-columns: 1fr; }}
  .f1-pcard.p1, .f1-pcard.p2, .f1-pcard.p3 {{ order: 0; min-height: 0; }}
  .f1-pcard.p1 {{ order: -1; }}
  .f1-tower .hd, .f1-tower .r {{
    grid-template-columns: 22px 4px 44px minmax(70px, 1fr) var(--real, 0); gap: .45rem;
  }}
  .f1-tower .n, .f1-tower .m {{ display: none; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
  .f1-circuit .trace {{ stroke-dashoffset: 0; }}
}}
"""


def apply_theme() -> None:
    """Inyecta el CSS del tema en la página."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Aplica la paleta del tema a un gráfico Altair."""
    return (
        chart.configure(background="transparent", font=MONO)
        .configure_axis(
            labelColor=MUTED,
            titleColor=TEXT,
            gridColor=LINE,
            domainColor=LINE,
            tickColor=LINE,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            labelFont=MONO,
            titleFont=MONO,
        )
        .configure_legend(
            labelColor=TEXT, titleColor=TEXT, orient="top", labelFontSize=11
        )
        .configure_view(strokeWidth=0)
    )
