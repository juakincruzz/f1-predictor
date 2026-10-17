#!/usr/bin/env python3
"""Descarga temporadas concretas; pensado para ejecutarse en background.

Uso:
    nohup python3 scripts/download_years.py 2024 2025 2026 > data/download.log 2>&1 &
    python3 scripts/download_years.py 2019 --source fastf1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collect import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga temporadas concretas.")
    parser.add_argument("years", nargs="+", type=int)
    parser.add_argument("--source", choices=["jolpica", "fastf1"], default="jolpica")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.years, source=args.source, force=args.force)
