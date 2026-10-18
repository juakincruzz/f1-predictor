# 🏎️ F1 Podium Predictor

Predicción de podios en Fórmula 1 usando XGBoost multiclase con validación temporal.

## Objetivo

Entrenar un clasificador de gradient boosting para predecir quién acabará en el podio (P1, P2, P3) usando décadas de datos históricos de F1, feature engineering avanzado y validación walk-forward por temporadas, y predecir el podio de **la próxima carrera** del calendario.

## Stack

- **Datos**: [Jolpica](https://github.com/jolpica/jolpica-f1) (sucesora de Ergast, por defecto) y [FastF1](https://docs.fastf1.dev/) como alternativa (2010–hoy)
- **Modelo**: XGBoost (`multi:softprob`) con SHAP nativo para explicar cada carrera
- **Dashboard**: Streamlit + Altair, estética de muro de boxes con colores de sector F1
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

`make dashboard` abre la app con estética de muro de boxes y tres secciones arriba (la sección, la temporada y la ronda quedan en la URL, así que puedes recargar o compartir sin perder dónde estabas):

- **Carrera**: eliges temporada y Gran Premio en una tira de calendario. Cada ronda disputada se colorea según cuántos pilotos del podio previsto acabaron en el podio (morado 3/3, verde 2/3, amarillo 1/3, rojo 0/3), y la próxima va marcada con ▸. Una línea «Viendo» indica siempre qué carrera tienes abierta. Debajo hay tres pestañas:
  - **Predicción**: en la próxima carrera, cabecera con semáforo de salida y cuenta atrás, podio previsto, trazado del circuito y torre de probabilidades. Si la clasificación ya está publicada usa la parrilla real; si no, la estima con la media de las últimas 5 qualys de cada piloto. En una carrera disputada, predicción frente a resultado real, aciertos de podio y una columna «Real» en la torre.
  - **Repetición**: repetición animada de la carrera (desde 2023) con los coches moviéndose por el circuito, torre de posiciones con gaps, neumáticos y paradas, estado de pista (amarilla, SC, VSC, roja) y mensajes de dirección de carrera. Controles de play/pausa, velocidad x1–x60 y barra de tiempo. La primera vez descarga la carrera de [OpenF1](https://openf1.org) (~1-2 min, plan gratuito) y la deja en caché en `data/openf1/`.
  - **Por qué**: SHAP de la carrera, es decir, qué variables movieron la probabilidad de podio.
- **Temporada**: balance del modelo ronda a ronda (aciertos de podio, ganadores acertados, podios perfectos) y acceso directo a cualquier carrera.
- **Modelo**: métricas walk-forward fuera de muestra por temporada e importancia global de las variables.

Los colores de sector significan lo mismo en todo el dashboard: morado = lo mejor (favorito, acierto exacto), verde = en el podio, amarillo = aviso o sorpresa, rojo = fallo. Las animaciones se desactivan si el sistema tiene activado «reducir movimiento».

El botón **Actualizar datos** descarga los últimos resultados de la temporada actual sin salir de la app.

## Features principales

Todas las features históricas solo usan carreras **anteriores** a la que se predice (`shift(1)` dentro de cada grupo) y siguen al piloto por `DriverId`, no por dorsal.

- `grid_position`: posición de salida (la de la parrilla real, con penalizaciones; si falta, la de la qualy)
- `recent_form_avg`: media de las posiciones finales del piloto en sus últimas 5 carreras
- `team_pace_avg`: media de las posiciones finales de la escudería en las últimas 5 carreras
- `track_history_avg`: media del piloto en ese circuito en sus últimas 5 visitas. Se agrupa por `CircuitId`, no por el nombre del GP, que cambia con los patrocinadores y a veces de circuito
- `championship_position`: posición en el mundial antes de la carrera, con los puntos reales (incluye la vuelta rápida)
- `races_experience`: carreras disputadas en toda su trayectoria
- `teammate_gap`: su resultado en la carrera anterior menos el de su compañero en la suya
- `race_round`: ronda de la temporada normalizada con la longitud real del calendario

Los NaN (un debutante sin historial, por ejemplo) se rellenan con las medianas del entrenamiento, que van guardadas dentro de `models/xgb_model.json`.

## Resultados

Validación walk-forward: cada temporada se predice con un modelo entrenado solo con las anteriores.

| Temporada | Accuracy | F1 macro | Log loss |
|---|---|---|---|
| 2023 | 0.864 | 0.502 | 0.457 |
| 2024 | 0.858 | 0.395 | 0.533 |
| 2025 | 0.873 | 0.485 | 0.410 |

La accuracy es alta en parte porque el 86 % de las filas son «sin podio»; el F1 macro y el log loss reflejan mejor cuánto acierta con los podios. El detalle está en `reports/` (`make evaluate`).

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
- ✅ Feature engineering sin fugas de datos (`shift(1)` por piloto con `DriverId`, circuito por `CircuitId`, medianas de entrenamiento guardadas en el modelo)
- ✅ XGBoost multiclase con early stopping y reentreno con todas las temporadas
- ✅ Validación temporal walk-forward (2023–2025)
- ✅ Predicción de la próxima carrera
- ✅ Dashboard Streamlit «muro de boxes»: calendario navegable, cuenta atrás, torre de probabilidades, balance de temporada y SHAP
- ✅ Trazado del circuito y repetición animada de carreras (OpenF1)
- ✅ CI/CD con GitHub Actions
- ✅ Tests con pytest (sin red)

## Licencia

MIT. Los trazados de circuitos (`src/dashboard/assets/f1-circuits.geojson`) son de [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) (MIT, ver `src/dashboard/assets/LICENSE-f1-circuits.md`).

> **Datos en directo**: el plan gratuito de OpenF1 solo sirve sesiones terminadas. Con el plan de pago basta con exportar `OPENF1_TOKEN`; el cliente lo envía automáticamente.
