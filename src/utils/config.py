"""Módulo de configuración centralizada del proyecto."""

from pathlib import Path

import yaml


class Config:
    """Clase singleton para cargar y acceder a la configuración del proyecto."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Carga el archivo config.yaml desde la raíz del proyecto."""
        root = Path(__file__).resolve().parent.parent.parent
        config_path = root / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró config.yaml en {root}")
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Obtiene un valor de configuración por clave usando dot notation.

        Ejemplo: config.get('data.start_year') -> 2010
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __getitem__(self, key):
        return self.get(key)

    @property
    def raw(self):
        """Devuelve la ruta absoluta a data/raw."""
        return Path(self.get("data.raw_path")).resolve()

    @property
    def processed(self):
        """Devuelve la ruta absoluta a data/processed."""
        return Path(self.get("data.processed_path")).resolve()

    @property
    def models(self):
        """Devuelve la ruta absoluta a models/."""
        return Path(self.get("paths.models")).resolve()

    @property
    def reports(self):
        """Devuelve la ruta absoluta a reports/."""
        return Path(self.get("paths.reports")).resolve()


config = Config()
