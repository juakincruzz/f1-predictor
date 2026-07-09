#!/usr/bin/env python3
"""Script independiente para descargar datos F1 año por año.
Se puede ejecutar con nohup y sobrevive a desconexiones.
Uso: nohup python3 scripts/download_years.py > data/download.log 2>&1 &
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect import collect_year

# Años faltantes o incompletos
YEARS_TO_DOWNLOAD = [2018, 2019, 2022, 2024, 2025, 2026]

if __name__ == "__main__":
    for year in YEARS_TO_DOWNLOAD:
        print(f"\n{'='*60}")
        print(f"DESCARGANDO AÑO {year}")
        print(f"{'='*60}")
        try:
            stats = collect_year(year, session_delay=10)
            print(f"Año {year} completado: {stats}")
        except Exception as e:
            print(f"ERROR en año {year}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuando con el siguiente año...")
            continue
    
    print(f"\n{'='*60}")
    print("DESCARGA FINALIZADA")
    print(f"{'='*60}")
