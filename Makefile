VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install test build clean run deps

run:
	$(PYTHON) localllmrag/localllmrag.py

venv:
	python3 -m venv $(VENV_DIR)

deps: requirements.txt
	$(PIP) install -r requirements.txt

install: venv deps
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests

build:
	$(PIP) install build setuptools
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info