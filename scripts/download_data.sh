#!/usr/bin/env bash
# Script de descarga de datos históricos en background
# Guarda logs en data/download.log

cd "$(dirname "$0")" || exit 1
source venv/bin/activate

echo "$(date): Iniciando descarga de datos F1..."

# Descargar 2018-2026 con rate limiting
python -c "
from src.data.collect import collect_range
collect_range(2018, 2026, session_delay=10)
"

echo "$(date): Descarga completada."
