VENV ?= .venv
PYTHON ?= python

ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
else
  VENV_BIN := $(VENV)/bin
endif

PY := $(VENV_BIN)/python
BENCH := $(VENV_BIN)/rag-eval-bench

.PHONY: install lint format test bench smoke serve docker-build docker-run clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

test:
	$(PY) -m pytest -q

bench:
	$(BENCH) run --retriever bm25 --k 5

smoke:
	$(BENCH) run --retriever bm25 --k 5 --name smoke

serve:
	$(BENCH) serve --reload

docker-build:
	docker build -t rag-eval-bench:local .

docker-run:
	docker run --rm -p 8000:8000 rag-eval-bench:local

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
