.PHONY: help install test dashboard evaluate clean lint

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
STREAMLIT ?= $(if $(wildcard $(VENV)/bin/streamlit),$(VENV)/bin/streamlit,streamlit)
PYTEST ?= $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)

help:
	@echo "Civil Unrest Event Extraction & Prediction"
	@echo "Available commands:"
	@echo "  make install      - Install dependencies from requirements.txt"
	@echo "  make test         - Run test suite with pytest"
	@echo "  make dashboard    - Launch the Streamlit forecast dashboard"
	@echo "  make evaluate     - Run 2024 test evaluation across models"
	@echo "  make clean        - Remove Python cache files and build artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

test:
	PYTHONPATH=. $(PYTEST) tests/ -v

dashboard:
	$(STREAMLIT) run app.py

evaluate:
	PYTHONPATH=. $(PYTHON) backend/evaluate_dataset.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
