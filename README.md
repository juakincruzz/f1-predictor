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

## Instalación

```bash
make install
```

## Uso

```bash
# Descargar datos (incremental)
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

## Features principales

- `grid_position`: Posición de salida
- `recent_form_avg`: Media últimas 5 carreras
- `team_pace_avg`: Ritmo de la escudería
- `track_history_avg`: Historial en el circuito
- `championship_position`: Posición en el mundial
- `races_experience`: Carreras acumuladas
- `teammate_gap`: Diferencia con el compañero

## Git Workflow

- `main`: Rama estable
- `dev`: Desarrollo e integración
- `feature/*`: Nuevas funcionalidades (PR a `dev`)

## Licencia

MIT
