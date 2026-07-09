"""Tests básicos para el módulo de datos."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.collect import get_raw_path
from src.utils.config import Config


def test_config_loads():
    config = Config()
    assert config.get("data.start_year") == 2010


def test_raw_path_exists():
    path = get_raw_path("2010")
    assert path.exists()


def test_placeholder():
    """Placeholder hasta tener datos reales para tests más robustos."""
    assert True
