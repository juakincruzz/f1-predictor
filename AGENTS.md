# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

F1 Podium Predictor: an XGBoost multiclass classifier that predicts F1 podium finishes (P1/P2/P3/no podium) using historical race data (2010–2026) from Jolpica (Ergast successor, default) or FastF1, with walk-forward temporal validation, next-race prediction and a Streamlit dashboard.

## Commands

```bash
make install     # pip install -r requirements.txt
make data        # python -m src.data.collect — incremental download (Jolpica by default; --source fastf1, --years, --force)
make update      # download only the current season
make features    # python -m src.features.build_features — builds data/processed/features.parquet
make train       # python -m src.models.train — trains XGBoost (early stopping on val_year, then refit on all seasons), saves models/xgb_model.json
make evaluate    # python -m src.models.evaluate — walk-forward validation, writes reports/
make dashboard   # streamlit run src/dashboard/app.py
make lint        # ruff check src tests && black --check src tests
make lint-fix    # ruff check --fix src tests && black src tests
make test        # pytest tests/ -v
make clean       # remove __pycache__, .pyc, .pytest_cache, .ruff_cache
```

Run a single test: `pytest tests/test_features.py::test_build_targets -v`

CI (`.github/workflows/ci.yml`) runs ruff, black --check, and pytest on push/PR to `main` and `dev` using Python 3.11. Run `make lint` and `make test` before committing.

## Pipeline order

The stages are strictly sequential and each reads the previous stage's output from disk (no in-memory chaining across `make` targets): `data` → `features` → `train` → `evaluate` / `dashboard`. If raw parquet files are missing, `features` produces an empty frame for that year; if `data/processed/features.parquet` is missing, `train` raises `FileNotFoundError` telling you to run `make features` first.

## Architecture

- **`src/utils/config.py`** — `Config` singleton loading `config.yaml` from the project root; access via the shared `config` instance with dot-notation keys (`config.get("data.start_year")`). All hyperparameters, feature lists, date ranges, and validation splits live in `config.yaml`, not in code.
- **`src/utils/paths.py`** — path helpers (`get_raw_path`, `get_processed_path`, `get_models_path`, `get_reports_path`) that resolve relative to `PROJECT_ROOT` and auto-create directories. Always use these instead of hardcoding paths.
- **`src/data/jolpica.py`** — default data source. Downloads schedule, race results and qualifying per season from the Jolpica API (a few paginated calls per season; pagination is by row, so `_merge_race` stitches races split across pages). Writes the **same schema as FastF1's `session.results`** to `data/raw/{year}/R{round:02d}_{R|Q}.parquet`, plus `data/raw/{year}/schedule.parquet`. `TEAM_NAMES` maps `constructorId` to FastF1's team names so team features don't break when sources mix; `TEAM_COLORS` feeds the dashboard. Incremental: only missing rounds are written.
- **`src/data/collect.py`** — CLI entry point (`main`/`run`: `--source {jolpica,fastf1}`, `--years`, `--force`) and `load_schedule(year)`. Also holds the FastF1 collector: `session.load(laps=False, telemetry=False, weather=False, messages=False)` (only `session.results` is used — keep the load lightweight), caching to `data/cache`, skipping existing parquets unless `force=True`, with a delay between sessions and exponential-backoff retries on `RateLimitExceededError` (~500 calls/hour).
- **`src/features/build_features.py`** — turns raw per-session parquet into the modeling table. Two invariants to preserve when adding features: (1) historical features (`recent_form_avg`, `team_pace_avg`, `track_history_avg`, etc.) group drivers by `driver_key(df)` (`DriverId`, falling back to `DriverNumber` — numbers change across seasons) and use per-group `transform` with `.shift(1)` *inside* the lambda so a race's features never leak that race's own result and rolling windows never cross group boundaries; (2) `build_features_from_frames` computes everything over the full multi-season frame so history carries across years — `build_features_for_year(year)` (used by the dashboard) loads all seasons up to `year` and filters, it does NOT build the year in isolation. `grid_position` prefers the race result's `GridPosition` (includes penalties) over qualifying position. Output columns come from `config.yaml`'s `features.columns` list, so new features must be added both here and there. Writes `data/processed/features.parquet`.
- **`src/models/train.py`** — `prepare_xy(df, medians=None)` selects only the `features.columns` from `config.yaml` (extra columns like `FullName`/`TeamColor` never reach the model) and fills NaNs with the given medians (pass train medians for eval sets to avoid leakage); `train_model` uses a *temporal* split (train = years < `validation.val_year`, early-stopping eval = that year; falls back to no-early-stopping if the val year has no data). Has a `try/except TypeError` fallback because `early_stopping_rounds` moved from `.fit()` to the constructor between XGBoost 2.x and 3.x — keep both paths if touching this. `best_iteration` is read via `getattr(..., None)` because it raises `AttributeError` without early stopping. `train_model(df, test_size=0.0)` (no holdout) is what `evaluate.py` uses for walk-forward folds; `train_model(df, refit_full=True)` (used by `__main__`) refits on every season with `best_iteration + 1` trees so the deployed model knows the latest seasons.
- **`src/models/evaluate.py`** — walk-forward validation with an expanding window: for each season in `validation.test_years`, trains on all seasons strictly before it. Writes `reports/walk_forward_results.csv`, `reports/walk_forward_summary.json`, and `reports/metrics_by_year.png`.
- **`src/models/predict.py`** — loads the saved model and maps class indices to labels via `CLASS_LABELS = {0: "No Podium", 1: "P1", 2: "P2", 3: "P3"}`; `predict_for_race` filters to one `(year, round)` and ranks by summed P1+P2+P3 probability (`PodiumProbability`), not by the raw multiclass argmax.
- **`src/models/upcoming.py`** — next-race prediction. `next_race` picks the first schedule round without results (skipping rounds more than `STALE_DAYS` past, i.e. cancelled); `build_upcoming_entries` uses that round's qualifying as the real grid if it exists, else the last race's entry list with a grid estimated from each driver's last 5 qualifyings (no history → back of the grid). The rows are appended to the history with no `Position` and passed through `build_features_from_frames`, so `shift(1)` features only see past races.
- **`src/models/explain.py`** — `feature_importance` (gain, normalized) and `podium_shap` (XGBoost native `pred_contribs` TreeSHAP, mean |SHAP| summed over P1–P3). The `shap` library is not used.
- **`src/dashboard/`** — `app.py` (tabs: Próxima carrera, Predicción, Histórico, Features; sidebar season/GP pickers and a button that runs `collect_year_jolpica` for the current season), `theme.py` (F1 red/silver palette, CSS, `style_chart` for Altair), `components.py` (escaped HTML cards/podium and the probability table). Streamlit theme lives in `.streamlit/config.toml`. Streamlit 1.59 tabs are `[data-testid="stTab"]` (react-aria), not `data-baseweb`. `tests/test_dashboard.py` smoke-tests the app with `AppTest`.
- **`scripts/download_years.py`** — thin argparse wrapper over `collect.run` for background downloads (`nohup ... 2024 2025 2026 &`).

All modules under `src/` insert the project root onto `sys.path` at import time and use absolute imports (`from src.utils.config import config`), and are runnable both as `python -m src.<pkg>.<module>` and imported directly — preserve this when adding new modules.

## Target encoding

Podium classes are fixed across the codebase: `0 = No Podium, 1 = P1, 2 = P2, 3 = P3` (see `TARGET_MAP`/`NO_PODIUM` in `build_features.py` and `CLASS_LABELS` in `predict.py`). `model.num_class` in `config.yaml` must stay at 4 to match.

## Git workflow

- `main`: stable/production branch.
- `dev`: integration branch.
- `feature/*`: new work, PR'd into `dev`; `dev` is periodically PR'd into `main`.
