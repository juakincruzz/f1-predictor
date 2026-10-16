# Rediseño rojo/plata y cierre del proyecto — Diseño

Fecha: 2026-10-16 · Rama: `feature/redesign-finish`

## Objetivo

Dejar F1 Podium Predictor terminado y usable: dashboard con estética clásica de F1
(rojo Fórmula + gris plateado), predicción de la próxima carrera real, datos
recientes (2024–2026) y repo limpio (lint/tests verdes, README al día).

Fuera de alcance: completar 2011–2017/2022, datos en vivo, despliegue.

## 1. Datos (Jolpica)

- `src/data/jolpica.py`: cliente de la API Jolpica (sucesora de Ergast,
  `https://api.jolpi.ca/ergast/f1`). Por temporada descarga:
  - calendario (`/{year}.json`) → `data/raw/{year}/schedule.parquet`
    (`RoundNumber`, `EventName`, `EventDate`, `Country`, `Locality`);
  - resultados de carrera (`/{year}/results.json`, paginado `limit=100`);
  - clasificación (`/{year}/qualifying.json`, paginado).
- Se escriben con el **esquema FastF1** en `data/raw/{year}/R{nn}_{R|Q}.parquet`:
  `DriverNumber` (str), `Abbreviation`, `DriverId`, `FullName`, `TeamName`,
  `TeamId`, `TeamColor`, `Position` (float), `ClassifiedPosition`,
  `GridPosition` (float, solo R), `Status`, `Points`, `Year`, `RoundNumber`,
  `SessionType`, `EventName`.
- `TeamName` se normaliza a los nombres de FastF1 vía `constructorId`
  (tabla `TEAM_NAMES`), con fallback al nombre de Jolpica. `TeamColor` sale de
  una tabla `TEAM_COLORS` (fallback plata).
- Incremental: salta parquets existentes salvo `force=True`. Pausa corta entre
  llamadas y reintento con backoff ante HTTP 429/5xx.
- `src/data/collect.py` gana CLI: `--source {jolpica,fastf1}` (defecto jolpica),
  `--years`, `--force`. `scripts/download_years.py` recibe años por argumento.

## 2. Modelo

- `config.yaml`: `validation.val_year: 2025`.
- `train_model`: tras early stopping en `val_year`, si `refit_full` (nuevo,
  defecto `True` en `__main__`) reentrena con todas las temporadas usando
  `best_iteration + 1` árboles. Walk-forward sigue usando `test_size=0`.
- Se regeneran features, modelo y reports (test_years 2023–2025).

## 3. Próxima carrera

`src/models/upcoming.py`:
- `next_race(schedule, race_rounds, today)` → primera ronda del calendario sin
  parquet de carrera (y con fecha ≥ hoy − 3 días, para no tomar carreras
  canceladas antiguas). `None` si la temporada terminó.
- `build_upcoming_entries(race_hist, quali_hist, year, round, event_name)` →
  pilotos/equipos de la última carrera disputada; `GridPosition` = quali real
  si existe `R{nn}_Q.parquet`, si no estimación: ranking de la media de
  posiciones de quali del piloto en sus últimas 5 apariciones (sin historial →
  al final). Devuelve también `grid_source ∈ {"real", "estimada"}`.
- `predict_next_race()` → concatena las entradas (sin `Position`) al histórico,
  llama a `build_features_from_frames`, filtra la ronda y aplica
  `predict_for_race`. Gracias a `shift(1)` las features no ven la carrera.

## 4. Dashboard

- `.streamlit/config.toml`: tema oscuro; primary `#E10600`, fondo `#0B0B0D`,
  secundario `#17181C`, texto `#D9D9DE`.
- `src/dashboard/theme.py`: CSS (Titillium Web, cabecera plata cepillada con
  franja roja, tarjetas con borde cromado, tabs rojo/plata) y paleta para gráficos.
- `src/dashboard/components.py`: `header()`, `podium_cards(df)`, `stat_card()`,
  tablas con formato porcentaje.
- `src/dashboard/app.py`: sidebar (temporada con datos, GP por nombre); tabs
  🏁 Próxima carrera · 🔮 Predicción (vs resultado real, aciertos x/3) ·
  📊 Histórico (Altair) · 🧠 Features (importancia XGBoost + SHAP medio de la
  carrera). Sin "Modo" ni "En Vivo". `st.cache_data` / `st.cache_resource`.

## 5. Limpieza y verificación

- Tests sin red: mapeo Jolpica → esquema (JSON fixture), `next_race`,
  `build_upcoming_entries` (grid real y estimada), refit.
- `ruff`, `black --check`, `pytest` verdes; arranque del dashboard sin errores
  (`streamlit run` headless + petición HTTP).
- README y CLAUDE.md/AGENTS.md actualizados; commitear AGENTS.md.
