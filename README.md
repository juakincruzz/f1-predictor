# 🏎️ F1 Podium Predictor

Predicción de podios en Fórmula 1 usando XGBoost multiclase con validación temporal.

## Objetivo

Entrenar un clasificador de gradient boosting para predecir quién acabará en el podio (P1, P2, P3) usando décadas de datos históricos de F1, feature engineering avanzado y validación walk-forward por temporadas, y predecir el podio de **la próxima carrera** del calendario.

## Stack

- **Datos**: [Jolpica](https://github.com/jolpica/jolpica-f1) (sucesora de Ergast, por defecto) y [FastF1](https://docs.fastf1.dev/) como alternativa (2010–hoy)
- **Modelo**: XGBoost (`multi:softprob`) con SHAP nativo para explicar cada carrera
- **Dashboard**: Streamlit + Altair, estética F1 clásica (rojo Fórmula y plata)
- **Pipeline**: Makefile + scripts Python
- **CI/CD**: GitHub Actions (ruff, black, pytest)

## Estructura

```
f1-predictor/
├── data/               # Datos crudos y procesados (incremental)
├── notebooks/          # Exploración, features, modelo, evaluación
├── src/
│   ├── data/           # Recolección (Jolpica / FastF1)
│   ├── features/       # Feature engineering
│   ├── models/         # Train, predict, evaluate, upcoming, explain
│   ├── dashboard/      # App Streamlit (app, theme, components)
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
# Descargar datos (incremental, vía Jolpica)
make data

# Actualizar solo la temporada en curso (p. ej. tras cada GP o qualy)
make update

# Generar features
make features

# Entrenar modelo
make train

# Evaluar con validación temporal
make evaluate

# Lanzar dashboard
make dashboard
```

> **Nota**: `make data` descarga todas las temporadas de `config.yaml` (2010–2026) desde Jolpica: unas pocas llamadas por temporada, así que tarda minutos y no horas. Es incremental: solo baja las rondas que faltan.

Opciones de descarga:

```bash
python -m src.data.collect --years 2024 2025 2026     # temporadas concretas
python -m src.data.collect --years 2019 --source fastf1   # usando FastF1
nohup python3 scripts/download_years.py 2011 2012 > data/download.log 2>&1 &   # en background
```

Predicción de la próxima carrera desde la terminal:

```bash
python -m src.models.upcoming
```

## Dashboard

`make dashboard` abre la app con cuatro pestañas:

- **🏁 Próxima carrera**: podio previsto y probabilidades para el siguiente GP del calendario. Si la clasificación ya está publicada usa la parrilla real; si no, la estima con la media de las últimas 5 qualys de cada piloto.
- **🔮 Predicción**: cualquier carrera disputada, predicción frente a resultado real y aciertos de podio.
- **📊 Histórico**: métricas walk-forward fuera de muestra por temporada.
- **🧠 Features**: importancia global del modelo y SHAP de la carrera seleccionada.

El botón **Actualizar temporada actual** de la barra lateral descarga los últimos resultados sin salir de la app.

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
# Cuando dev está estable: PR dev → main
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

## Estado

- ✅ Recolección incremental con Jolpica (por defecto) o FastF1
- ✅ Feature engineering sin fugas de datos (`shift(1)` por piloto con `DriverId`)
- ✅ XGBoost multiclase con early stopping y reentreno con todas las temporadas
- ✅ Validación temporal walk-forward (2023–2025)
- ✅ Predicción de la próxima carrera
- ✅ Dashboard Streamlit rojo/plata con SHAP
- ✅ CI/CD con GitHub Actions
- ✅ Tests con pytest (sin red)

## Licencia

MIT
