VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(PYTHON) -m pip
PYTHONPATH := src

.PHONY: venv install test build clean run deps build-release build-deps dev-deps

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m localllmrag.localllmrag

watch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src/localllmrag.re-indexer

venv:
	python3 -m venv $(VENV_DIR)

build-deps:
	$(PIP) install build setuptools

dev-deps: deps requirements-dev.txt
	$(PIP) install -r requirements-dev.txt

deps: requirements.txt
	$(PIP) install -r requirements.txt

ruff:
	$(PYTHON) -m ruff check --output-format=github src/.

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests

__version__:
	bash scripts/setversion.sh

build-release: __version__ build

build: build-deps
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info __version__