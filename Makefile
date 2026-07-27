# Makefile for U.S. REIT Multi-Agent LLM Project
# See CLAUDE.md Section 8 for target descriptions.

PYTHON ?= python3
PYTEST ?= pytest
CONFIG ?= config/config.yaml

.PHONY: install reproduce_v6 test lint prompt_sha llm_dev_run llm_validation llm_test_run

# ---------- Setup ----------

install:
	pip install -r requirements.txt

# ---------- Regression ----------

reproduce_v6:
	@echo "=== reproduce_v6: re-run V6 baseline and diff vs golden snapshots ==="
	@echo "Note: re-runs 4-fold purged CV + hyperparameter search; may take several minutes."
	$(PYTHON) tests/test_v6_reproduction.py

# ---------- Testing ----------

test:
	$(PYTEST) tests/ -v

# ---------- Linting ----------

lint:
	ruff check src/ tests/ llm/ || true
	black --check src/ tests/ llm/ || true

# ---------- Prompts ----------

prompt_sha:
	@$(PYTHON) -c "\
	import hashlib; \
	from pathlib import Path; \
	for p in sorted(Path('llm/prompts').glob('*.md')): \
	    body = p.read_text(); \
	    sha = hashlib.sha256(body.encode()).hexdigest()[:16]; \
	    print(f'{p.name}: {sha}')"

# ---------- LLM Runs ----------

llm_dev_run:
	$(PYTHON) -m llm.run --mode dev

llm_validation:
	$(PYTHON) -m llm.run --mode validation

llm_test_run:
	$(PYTHON) -m llm.run --mode test

# ---------- Reporting ----------

