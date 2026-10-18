"""Tests para feature engineering."""

import pandas as pd

from src.features.build_features import (
    add_race_round,
    build_features_from_frames,
    build_targets,
    compute_championship_position,
    compute_races_experience,
    compute_recent_form,
    compute_teammate_gap,
    compute_track_history,
    driver_key,
    merge_race_and_quali,
)


def test_build_targets():
    df = pd.DataFrame(
        {
            "Position": [1, 2, 3, 4, 5],
            "DriverNumber": ["44", "1", "16", "11", "33"],
        }
    )
    result = build_targets(df)
    assert "target" in result.columns
    assert result["target"].tolist() == [1, 2, 3, 0, 0]


def test_recent_form_no_cross_driver_contamination():
    """La media móvil de cada piloto solo debe usar SUS carreras anteriores."""
    df = pd.DataFrame(
        {
            "Year": [2023] * 6,
            "RoundNumber": [1, 1, 2, 2, 3, 3],
            "DriverNumber": ["44", "1", "44", "1", "44", "1"],
            "final_position": [1, 10, 2, 9, 3, 8],
        }
    )
    result = compute_recent_form(df, n=5)

    d44_r3 = result[(result["DriverNumber"] == "44") & (result["RoundNumber"] == 3)]
    d1_r3 = result[(result["DriverNumber"] == "1") & (result["RoundNumber"] == 3)]

    # Piloto 44: media de sus posiciones previas (1, 2) = 1.5
    assert d44_r3["recent_form_avg"].iloc[0] == 1.5
    # Piloto 1: media de sus posiciones previas (10, 9) = 9.5
    assert d1_r3["recent_form_avg"].iloc[0] == 9.5


def test_track_history_no_cross_driver_contamination():
    """El historial en un circuito debe ser del piloto, no mezclado con otros."""
    df = pd.DataFrame(
        {
            "Year": [2022, 2022, 2023, 2023],
            "RoundNumber": [1, 1, 1, 1],
            "DriverNumber": ["44", "1", "44", "1"],
            "EventName": ["Bahrain Grand Prix"] * 4,
            "final_position": [1, 10, 2, 9],
        }
    )
    result = compute_track_history(df, n=5)

    d1_2023 = result[(result["DriverNumber"] == "1") & (result["Year"] == 2023)]
    # Piloto 1 en Bahrain 2023: su única aparición previa fue P10 en 2022
    assert d1_2023["track_history_avg"].iloc[0] == 10.0


def test_features_carry_across_seasons():
    """races_experience y track_history deben acumular entre temporadas."""
    race_df = pd.DataFrame(
        {
            "Year": [2022, 2022, 2022, 2023],
            "RoundNumber": [1, 2, 3, 1],
            "EventName": ["Bahrain GP", "Jeddah GP", "Melbourne GP", "Bahrain GP"],
            "DriverNumber": ["44"] * 4,
            "Abbreviation": ["HAM"] * 4,
            "TeamName": ["Mercedes"] * 4,
            "Position": [3, 5, 4, 2],
            "GridPosition": [2, 4, 3, 1],
        }
    )
    quali_df = pd.DataFrame()

    result = build_features_from_frames(race_df, quali_df)
    row_2023 = result[result["Year"] == 2023].iloc[0]

    # 3 carreras disputadas antes de la primera de 2023
    assert row_2023["races_experience"] == 3
    # Historial en Bahrain: P3 en 2022
    assert row_2023["track_history_avg"] == 3.0


def test_grid_position_prefers_race_gridposition():
    """GridPosition del resultado de carrera (con penalizaciones) manda sobre quali."""
    race_df = pd.DataFrame(
        {
            "Year": [2023],
            "RoundNumber": [1],
            "DriverNumber": ["44"],
            "GridPosition": [3.0],  # salió P3 por penalización
        }
    )
    quali_df = pd.DataFrame(
        {
            "Year": [2023],
            "RoundNumber": [1],
            "DriverNumber": ["44"],
            "Position": [1.0],  # hizo la pole en quali
        }
    )
    merged = merge_race_and_quali(race_df, quali_df)
    assert merged["grid_position"].iloc[0] == 3


def test_grid_position_falls_back_to_quali():
    """Si el resultado de carrera no trae GridPosition, usar la posición de quali."""
    race_df = pd.DataFrame(
        {
            "Year": [2023],
            "RoundNumber": [1],
            "DriverNumber": ["44"],
        }
    )
    quali_df = pd.DataFrame(
        {
            "Year": [2023],
            "RoundNumber": [1],
            "DriverNumber": ["44"],
            "Position": [5.0],
        }
    )
    merged = merge_race_and_quali(race_df, quali_df)
    assert merged["grid_position"].iloc[0] == 5


def test_recent_form_follows_driver_id_across_number_change():
    """Un piloto que cambia de dorsal mantiene su forma reciente."""
    df = pd.DataFrame(
        {
            "Year": [2025, 2026],
            "RoundNumber": [24, 1],
            "DriverNumber": ["1", "3"],
            "DriverId": ["max_verstappen", "max_verstappen"],
            "final_position": [2, 5],
        }
    )
    result = compute_recent_form(df, n=5)
    assert result.loc[result["Year"] == 2026, "recent_form_avg"].iloc[0] == 2
    exp = compute_races_experience(df)
    assert exp.loc[exp["Year"] == 2026, "races_experience"].iloc[0] == 1


def test_driver_key_falls_back_to_number():
    df = pd.DataFrame({"DriverNumber": ["44", "1"], "DriverId": ["hamilton", None]})
    assert driver_key(df).tolist() == ["hamilton", "1"]
    assert driver_key(df[["DriverNumber"]]).tolist() == ["44", "1"]


def test_race_round_uses_season_length_when_known():
    """En una temporada en curso, la ronda se normaliza con el calendario
    completo, no con la última ronda disputada."""
    df = pd.DataFrame(
        {
            "Year": [2025, 2026, 2026, 2026],
            "RoundNumber": [24, 17, 18, 18],
            "TotalRounds": [None, 23, 23, None],
        }
    )
    result = add_race_round(df)["race_round"].tolist()
    assert result == [1.0, 17 / 23, 18 / 23, 18 / 23]


def _team_frame(order=None):
    """Dos compañeros (A, B) en tres carreras; order permite barajar filas."""
    df = pd.DataFrame(
        {
            "Year": [2024] * 6,
            "RoundNumber": [1, 1, 2, 2, 3, 3],
            "DriverNumber": ["1", "2"] * 3,
            "DriverId": ["a", "b"] * 3,
            "TeamName": ["T"] * 6,
            "final_position": [1, 5, 3, 2, 2, 6],
        }
    )
    return df.iloc[order] if order is not None else df


def _gap(df, driver, rnd):
    row = df[(df["DriverId"] == driver) & (df["RoundNumber"] == rnd)]
    return row["teammate_gap"].iloc[0]


def test_teammate_gap_uses_previous_race_of_both_drivers():
    """Gap = mi resultado anterior - el resultado anterior de mi compañero."""
    result = compute_teammate_gap(_team_frame())
    assert pd.isna(_gap(result, "a", 1))
    assert _gap(result, "a", 2) == 1 - 5
    assert _gap(result, "b", 2) == 5 - 1
    assert _gap(result, "a", 3) == 3 - 2
    assert _gap(result, "b", 3) == 2 - 3


def test_teammate_gap_never_sees_current_or_future_results():
    """Cambiar el resultado de una carrera no puede mover su propio gap ni los
    anteriores, y el orden de las filas dentro de la carrera no importa."""
    base = compute_teammate_gap(_team_frame())
    changed = _team_frame()
    changed.loc[changed["RoundNumber"] == 3, "final_position"] = [20, 1]
    changed = compute_teammate_gap(changed)
    shuffled = compute_teammate_gap(_team_frame(order=[1, 0, 3, 2, 5, 4]))
    for driver in ("a", "b"):
        for rnd in (2, 3):
            assert _gap(changed, driver, rnd) == _gap(base, driver, rnd)
            assert _gap(shuffled, driver, rnd) == _gap(base, driver, rnd)


def test_championship_uses_real_points_and_follows_driver_id():
    """Usa la columna Points (vuelta rápida, etc.) y no se rompe si el piloto
    cambia de dorsal a mitad de temporada."""
    df = pd.DataFrame(
        {
            "Year": [2024] * 6,
            "RoundNumber": [1, 1, 2, 2, 3, 3],
            "DriverNumber": ["1", "2", "1", "2", "33", "2"],
            "DriverId": ["a", "b", "a", "b", "a", "b"],
            "final_position": [1, 2, 2, 1, 1, 2],
            # a gana R1 con vuelta rápida (26); con la tabla fija empatarían a 43
            "Points": [26, 18, 18, 25, 25, 18],
        }
    )
    result = compute_championship_position(df)
    r3 = result[result["RoundNumber"] == 3].set_index("DriverId")
    # antes de R3: a = 44, b = 43 (a corre R3 con otro dorsal)
    assert r3.loc["a", "championship_position"] == 1
    assert r3.loc["b", "championship_position"] == 2


def test_track_history_follows_circuit_not_event_name():
    """Un GP que cambia de nombre sigue siendo el mismo circuito, y un nombre
    que cambia de circuito no hereda el historial del anterior."""
    df = pd.DataFrame(
        {
            "Year": [2024, 2025, 2026, 2026],
            "RoundNumber": [1, 1, 1, 2],
            "DriverNumber": ["1"] * 4,
            "EventName": [
                "Malaysian Grand Prix",
                "Bahrain Grand Prix in Malaysia",
                "Spanish Grand Prix",
                "Barcelona Grand Prix",
            ],
            "CircuitId": ["sepang", "sepang", "madring", "catalunya"],
            "final_position": [4, 6, 1, 1],
        }
    )
    result = compute_track_history(df, n=5).set_index(["Year", "RoundNumber"])
    assert result.loc[(2025, 1), "track_history_avg"] == 4
    assert pd.isna(result.loc[(2026, 1), "track_history_avg"])


def _fake_project(monkeypatch, tmp_path):
    """Rutas de datos en tmp_path y features.parquet ya construido con 2024."""
    import os

    from src.features import build_features as bf

    raw = tmp_path / "raw"
    (raw / "2024").mkdir(parents=True)
    raw_file = raw / "2024" / "R01_R.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(raw_file)
    processed = tmp_path / "features.parquet"
    pd.DataFrame({"Year": [2023, 2024], "RoundNumber": [1, 1]}).to_parquet(processed)
    os.utime(raw_file, (1_000, 1_000))  # el crudo es anterior a las features

    monkeypatch.setattr(bf, "get_raw_path", lambda sub: raw / sub)
    monkeypatch.setattr(bf, "get_processed_path", lambda name: processed)
    monkeypatch.setattr(bf.config, "get", lambda key, default=None: 2024)
    return bf, raw_file


def test_features_for_year_reuse_fresh_processed_file(monkeypatch, tmp_path):
    bf, _ = _fake_project(monkeypatch, tmp_path)

    def no_rebuild(*_):
        raise AssertionError("no debía recalcular")

    monkeypatch.setattr(bf, "load_raw_range", no_rebuild)
    result = bf.build_features_for_year(2024)
    assert result["Year"].tolist() == [2024]


def test_features_for_year_rebuild_when_raw_is_newer(monkeypatch, tmp_path):
    import os
    import time

    bf, raw_file = _fake_project(monkeypatch, tmp_path)
    os.utime(raw_file, (time.time() + 60, time.time() + 60))  # crudo nuevo
    calls = []
    monkeypatch.setattr(
        bf, "load_raw_range", lambda *a: calls.append(a) or pd.DataFrame()
    )
    assert bf.build_features_for_year(2024).empty
    assert calls  # ha vuelto a leer los crudos
