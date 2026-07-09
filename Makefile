.PHONY: help install data features train dashboard lint test clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install     - Instala dependencias"
	@echo "  make data        - Descarga/carga datos crudos (incremental)"
	@echo "  make features    - Genera features procesadas"
	@echo "  make train       - Entrena el modelo XGBoost"
	@echo "  make evaluate    - Evalúa el modelo con validación temporal"
	@echo "  make dashboard   - Lanza la app Streamlit"
	@echo "  make lint        - Ejecuta ruff y black"
	@echo "  make test        - Ejecuta pytest"
	@echo "  make clean       - Limpia archivos temporales"

install:
	pip install -r requirements.txt

data:
	python -m src.data.collect

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

dashboard:
	streamlit run src/dashboard/app.py

lint:
	ruff check src tests
	black --check src tests

lint-fix:
	ruff check --fix src tests
	black src tests

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.egg-info' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
