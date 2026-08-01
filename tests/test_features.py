"""Tests para feature engineering."""

import pandas as pd

from src.features.build_features import (
    build_features_from_frames,
    build_targets,
    compute_recent_form,
    compute_track_history,
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
