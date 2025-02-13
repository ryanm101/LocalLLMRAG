VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install test build clean run deps build-release build-deps dev-deps

run:
	$(PYTHON) -m localllmrag.localllmrag

venv:
	python3 -m venv $(VENV_DIR)

build-deps:
	$(PIP) install build setuptools

dev-deps: deps requirements-dev.txt
	$(PIP) install -r requirements-dev.txt

deps: requirements.txt
	$(PIP) install -r requirements.txt

ruff:
	ruff check --output-format=github .

install: venv deps
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests

__version__:
	sh scripts/setversion.sh

build-release: __version__ build


build: build-deps
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info __version__