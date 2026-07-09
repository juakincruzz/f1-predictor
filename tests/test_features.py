"""Tests básicos para feature engineering."""

import pandas as pd
import pytest

from src.features.build_features import build_targets


def test_build_targets():
    df = pd.DataFrame({
        "Position": [1, 2, 3, 4, 5],
        "DriverNumber": ["44", "1", "16", "11", "33"],
    })
    result = build_targets(df)
    assert "Target" in result.columns
    assert result["Target"].tolist() == [1, 2, 3, 0, 0]
