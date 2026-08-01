# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

F1 Podium Predictor: an XGBoost multiclass classifier that predicts F1 podium finishes (P1/P2/P3/no podium) using historical race data (2010–2026) from FastF1, with walk-forward temporal validation and a Streamlit dashboard.

## Commands

```bash
make install     # pip install -r requirements.txt
make data        # python -m src.data.collect — incremental download via FastF1 (rate-limited)
make features    # python -m src.features.build_features — builds data/processed/features.parquet
make train       # python -m src.models.train — trains XGBoost, saves models/xgb_model.json
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
- **`src/data/collect.py`** — downloads race (`R`) and qualifying (`Q`) sessions per year/round via FastF1 with `session.load(laps=False, telemetry=False, weather=False, messages=False)` (only `session.results` is used — keep the load lightweight), caching to `data/cache` and writing one parquet per session to `data/raw/{year}/R{round:02d}_{session_type}.parquet`. Skips sessions whose parquet already exists unless `force=True`, so re-running `make data` is incremental. Enforces FastF1's ~500 calls/hour limit with a configurable delay between sessions and exponential-backoff retries on `RateLimitExceededError`.
- **`src/features/build_features.py`** — turns raw per-session parquet into the modeling table. Two invariants to preserve when adding features: (1) historical features (`recent_form_avg`, `team_pace_avg`, `track_history_avg`, etc.) use per-group `transform` with `.shift(1)` *inside* the lambda so a race's features never leak that race's own result and rolling windows never cross group boundaries; (2) `build_features_from_frames` computes everything over the full multi-season frame so history carries across years — `build_features_for_year(year)` (used by the dashboard) loads all seasons up to `year` and filters, it does NOT build the year in isolation. `grid_position` prefers the race result's `GridPosition` (includes penalties) over qualifying position. Output columns come from `config.yaml`'s `features.columns` list, so new features must be added both here and there. Writes `data/processed/features.parquet`.
- **`src/models/train.py`** — `prepare_xy(df, medians=None)` drops ID/label columns (module-level `DROP_COLS`) and fills NaNs with the given medians (pass train medians for eval sets to avoid leakage); `train_model` uses a *temporal* split (train = years < `validation.val_year`, early-stopping eval = that year; falls back to no-early-stopping if the val year has no data). Has a `try/except TypeError` fallback because `early_stopping_rounds` moved from `.fit()` to the constructor between XGBoost 2.x and 3.x — keep both paths if touching this. `best_iteration` is read via `getattr(..., None)` because it raises `AttributeError` without early stopping. `train_model(df, test_size=0.0)` (no holdout) is what `evaluate.py` uses for walk-forward folds.
- **`src/models/evaluate.py`** — walk-forward validation with an expanding window: for each season in `validation.test_years`, trains on all seasons strictly before it. Writes `reports/walk_forward_results.csv`, `reports/walk_forward_summary.json`, and `reports/metrics_by_year.png`.
- **`src/models/predict.py`** — loads the saved model and maps class indices to labels via `CLASS_LABELS = {0: "No Podium", 1: "P1", 2: "P2", 3: "P3"}`; `predict_for_race` filters to one `(year, round)` and ranks by summed P1+P2+P3 probability (`PodiumProbability`), not by the raw multiclass argmax.
- **`src/dashboard/app.py`** — Streamlit app with four tabs (Predicción, Histórico, Features, En Vivo); currently reads directly from `build_features_for_year` + `predict_for_race` and from `reports/walk_forward_results.csv`. The Features and En Vivo tabs are placeholders (no SHAP integration yet despite `shap` being in requirements).
- **`scripts/download_years.py`** — standalone long-running download helper meant to be run with `nohup`/background (`data/download.log`) since a full historical pull can exceed FastF1's rate limit window; it hardcodes a `YEARS_TO_DOWNLOAD` list of specific gap years rather than a full range.

All modules under `src/` insert the project root onto `sys.path` at import time and use absolute imports (`from src.utils.config import config`), and are runnable both as `python -m src.<pkg>.<module>` and imported directly — preserve this when adding new modules.

## Target encoding

Podium classes are fixed across the codebase: `0 = No Podium, 1 = P1, 2 = P2, 3 = P3` (see `TARGET_MAP`/`NO_PODIUM` in `build_features.py` and `CLASS_LABELS` in `predict.py`). `model.num_class` in `config.yaml` must stay at 4 to match.

## Git workflow

- `main`: stable/production branch.
- `dev`: integration branch.
- `feature/*`: new work, PR'd into `dev`; `dev` is periodically PR'd into `main`.
