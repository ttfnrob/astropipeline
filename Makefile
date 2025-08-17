# AstroAgent Pipeline Makefile

.PHONY: setup test hypotheses review plan execute peerreview report clean lint format reproduce help

# Python interpreter
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

help: ## Show this help message
	@echo "AstroAgent Pipeline - Available commands:"
	@echo
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the development environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Virtual environment created. Activate with: source $(VENV)/bin/activate"

setup-prod: ## Set up the production environment (no dev dependencies)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

setup-requirements: ## Set up using requirements.txt files
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

install: ## Install the package and dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	$(PYTHON_VENV) -m pytest tests/ -v --cov=astroagent --cov-report=html --cov-report=term

test-unit: ## Run only unit tests
	$(PYTHON_VENV) -m pytest tests/ -v -m "unit"

test-integration: ## Run only integration tests
	$(PYTHON_VENV) -m pytest tests/ -v -m "integration"

lint: ## Run linting tools
	$(PYTHON_VENV) -m flake8 astroagent/ tests/
	$(PYTHON_VENV) -m mypy astroagent/

format: ## Format code with black and isort
	$(PYTHON_VENV) -m black astroagent/ tests/
	$(PYTHON_VENV) -m isort astroagent/ tests/

format-check: ## Check if code is properly formatted
	$(PYTHON_VENV) -m black --check astroagent/ tests/
	$(PYTHON_VENV) -m isort --check-only astroagent/ tests/

# ============================================================================
# Agent Pipeline Commands
# ============================================================================

hypotheses: ## Generate new hypotheses (usage: make hypotheses TAGS="stellar dynamics" N=10)
	$(PYTHON_VENV) -m astroagent.orchestration.graph run HM --tags "$(TAGS)" --n $(N)

review: ## Review ideas from the ideas register
	$(PYTHON_VENV) -m astroagent.orchestration.graph run RV --from data/registry/ideas_register.csv

plan: ## Create experiment plan for an idea (usage: make plan IDEA=01J...)
	$(PYTHON_VENV) -m astroagent.orchestration.graph run ED --idea $(IDEA)

execute: ## Execute experiment for an idea (usage: make execute IDEA=01J...)
	$(PYTHON_VENV) -m astroagent.orchestration.graph run EX --idea $(IDEA)

peerreview: ## Peer review an executed idea (usage: make peerreview IDEA=01J...)
	$(PYTHON_VENV) -m astroagent.orchestration.graph run PR --idea $(IDEA)

report: ## Generate final report for an idea (usage: make report IDEA=01J...)
	$(PYTHON_VENV) -m astroagent.orchestration.graph run RP --idea $(IDEA)

# ============================================================================
# Pipeline Management
# ============================================================================

pipeline-status: ## Show status of all ideas in the pipeline
	$(PYTHON_VENV) -c "import pandas as pd; print(pd.read_csv('data/registry/ideas_register.csv')[['idea_id', 'title', 'status']].to_string())"

pipeline-summary: ## Show summary statistics of the pipeline
	$(PYTHON_VENV) -c "import pandas as pd; df = pd.read_csv('data/registry/ideas_register.csv'); print(df['status'].value_counts())"

# ============================================================================
# Data Management
# ============================================================================

init-registries: ## Initialize empty registry CSV files
	mkdir -p data/registry
	$(PYTHON_VENV) -m astroagent.data.init_registries

backup-data: ## Backup all data files
	tar -czf data-backup-$$(date +%Y%m%d_%H%M%S).tar.gz data/

clean-cache: ## Clean vector database cache
	rm -rf data/vectors/*

# ============================================================================
# Reproducibility and Validation
# ============================================================================

reproduce: ## Test reproducibility on a clean environment
	@echo "Testing reproducibility..."
	$(PYTHON_VENV) -m astroagent.validation.reproduce_check

validate-env: ## Validate that all required environment variables are set
	$(PYTHON_VENV) -m astroagent.validation.env_check

validate-data: ## Validate data registry consistency
	$(PYTHON_VENV) -m astroagent.validation.data_check

# ============================================================================
# Development and Utilities
# ============================================================================

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docs: ## Build documentation (if using Sphinx)
	@echo "Documentation generation not yet implemented"

serve-api: ## Start the FastAPI service (if enabled)
	$(PYTHON_VENV) -m uvicorn astroagent.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t astroagent:latest .

docker-run: ## Run Docker container
	docker run -it --rm -v $$(pwd)/data:/app/data astroagent:latest

# ============================================================================
# Default variables
# ============================================================================

TAGS ?= "stellar dynamics,galactic structure"
N ?= 5
IDEA ?= ""
