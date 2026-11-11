# Convenience targets. `make help` lists them.
.PHONY: help install run test lint typecheck seed migrate revision clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install dev dependencies
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements-dev.txt

run:  ## Start the dev server with hot reload
	.venv/bin/uvicorn app.main:app --reload --port 8000

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint and format-check
	.venv/bin/ruff check app tests scripts alembic
	.venv/bin/ruff format --check app tests scripts alembic

typecheck:  ## Static type check
	.venv/bin/mypy app

seed:  ## Load demo data (an org, a matter, three contracts)
	$(PY) -m scripts.seed

migrate:  ## Apply migrations
	.venv/bin/alembic upgrade head

revision:  ## Autogenerate a migration: make revision m="add x"
	.venv/bin/alembic revision --autogenerate -m "$(m)"

clean:  ## Remove caches and local databases
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.db *.db-wal *.db-shm
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
