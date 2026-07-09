"""Utilidades de rutas para el proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """Crea el directorio si no existe y devuelve la ruta."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_path(subdir: str | None = None) -> Path:
    """Devuelve la ruta a data/raw/, opcionalmente con subdirectorio."""
    path = PROJECT_ROOT / "data" / "raw"
    if subdir:
        path = path / subdir
    return ensure_dir(path)


def get_processed_path(filename: str | None = None) -> Path:
    """Devuelve la ruta a data/processed/, opcionalmente con fichero."""
    path = PROJECT_ROOT / "data" / "processed"
    ensure_dir(path)
    if filename:
        return path / filename
    return path


def get_models_path(filename: str | None = None) -> Path:
    """Devuelve la ruta a models/, opcionalmente con fichero."""
    path = PROJECT_ROOT / "models"
    ensure_dir(path)
    if filename:
        return path / filename
    return path


def get_reports_path(filename: str | None = None) -> Path:
    """Devuelve la ruta a reports/, opcionalmente con fichero."""
    path = PROJECT_ROOT / "reports"
    ensure_dir(path)
    if filename:
        return path / filename
    return path
