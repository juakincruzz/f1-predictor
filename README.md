# 🏎️ F1 Podium Predictor

Predicción de podios en Fórmula 1 usando XGBoost multiclase con validación temporal.

## Objetivo

Entrenar un clasificador de gradient boosting para predecir quién acabará en el podio (P1, P2, P3) usando décadas de datos históricos de F1, feature engineering avanzado y validación walk-forward por temporadas.

## Stack

- **Datos**: [FastF1](https://docs.fastf1.dev/) + Ergast API (2010–hoy)
- **Modelo**: XGBoost (`multi:softprob`)
- **Dashboard**: Streamlit
- **Pipeline**: Makefile + scripts Python
- **CI/CD**: GitHub Actions (ruff, black, pytest)

## Estructura

```
f1-predictor/
├── data/               # Datos crudos y procesados (incremental)
├── notebooks/          # Exploración, features, modelo, evaluación
├── src/
│   ├── data/           # Recolección con FastF1
│   ├── features/       # Feature engineering
│   ├── models/         # Train, predict, evaluate
│   ├── dashboard/      # App Streamlit
│   └── utils/          # Config y rutas
├── tests/              # pytest
├── models/             # Artefactos entrenados
├── reports/            # Métricas y gráficos
├── Makefile            # Pipeline reproducible
└── config.yaml         # Hiperparámetros y rutas
```

## Instalación y Setup

Requiere Python 3.11+.

```bash
# 1. Clonar el repo
git clone https://github.com/juakincruzz/f1-predictor.git
cd f1-predictor

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Descargar datos (incremental) — puede tardar la primera vez
make data

# Generar features
make features

# Entrenar modelo
make train

# Evaluar con validación temporal
make evaluate

# Lanzar dashboard
make dashboard
```

> **Nota**: `make data` descarga todas las temporadas configuradas en `config.yaml` (2010–2026). La primera ejecución puede tardar varios minutos. Las siguientes serán incrementales.

## Features principales

- `grid_position`: Posición de salida
- `recent_form_avg`: Media últimas 5 carreras por piloto
- `team_pace_avg`: Ritmo medio de la escudería reciente
- `track_history_avg`: Historial del piloto en el circuito
- `championship_position`: Posición acumulada en el mundial
- `races_experience`: Carreras acumuladas en F1
- `teammate_gap`: Diferencia con el compañero de equipo
- `race_round`: Número de carrera en la temporada (normalizado)

## Git Workflow

- `main`: Rama estable (producción)
- `dev`: Desarrollo e integración
- `feature/*`: Nuevas funcionalidades (PR a `dev`)

```bash
git checkout dev
git checkout -b feature/nueva-funcionalidad
# ... trabajas ...
git push -u origin feature/nueva-funcionalidad
# Abres PR en GitHub: feature/* → dev
g# Cuando dev está estable: PR dev → main
```

## Testeo

```bash
# Tests unitarios
make test

# Lint + format check
make lint

# Fix lint automáticamente
make lint-fix

# Clean
make clean
```

## Estado del MVP

- ✅ Recolección incremental con FastF1 (2010–2026)
- ✅ Feature engineering completo
- ✅ XGBoost multiclase entrenable
- ✅ Validación temporal walk-forward
- ✅ Dashboard Streamlit funcional
- ✅ CI/CD con GitHub Actions
- ✅ Tests base con pytest

## Licencia

MIT
