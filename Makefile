.PHONY: install test serve lint profile clean docker-build docker-run

PYTHON ?= python3
PIP ?= pip

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install ruff mypy pytest pytest-cov httpx

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

serve:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

profile:
	$(PYTHON) -c "\
	import pandas as pd; \
	from src.profiling.engine import ProfilingEngine; \
	df = pd.read_csv('tests/fixtures/sample.csv') if __import__('pathlib').Path('tests/fixtures/sample.csv').exists() else pd.DataFrame({'id': range(100), 'name': ['test']*100}); \
	result = ProfilingEngine().profile(df); \
	print(f'Rows: {result.row_count}, Columns: {result.column_count}'); \
	[print(f'  {n}: type={c.dtype}, nulls={c.null_pct}%, distinct={c.distinct_count}') for n, c in result.columns.items()]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache

docker-build:
	docker build -t data-quality-platform:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env data-quality-platform:latest
