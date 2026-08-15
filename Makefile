.PHONY: install lint typecheck test test-cov clean help demo demo-small validate benchmark report check

help:
	@echo "Available targets:"
	@echo "  install      - Install package with dev dependencies"
	@echo "  lint         - Run ruff linter"
	@echo "  typecheck    - Run mypy type checker"
	@echo "  test         - Run pytest test suite"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  demo-small   - Run 100-patient fast validation demo"
	@echo "  demo         - Run 1000-patient full SAD/MAD simulation"
	@echo "  validate     - Run benchmark harness (Warfarin + Moxifloxacin)"
	@echo "  benchmark    - Hardware acceleration comparison"
	@echo "  report       - Generate HTML/Markdown validation report"
	@echo "  clean        - Remove build artifacts"

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src/insilico_trial

typecheck:
	mypy src/insilico_trial

test:
	pytest src/insilico_trial/tests/ -x -q

test-cov:
	pytest src/insilico_trial/tests/ -x -q --cov=insilico_trial --cov-report=term-missing

demo-small:
	python -m insilico_trial.cli demo --patients 100 --duration-days 7

demo:
	python -m insilico_trial.cli demo --patients 1000 --duration-days 7

validate:
	python -m insilico_trial.cli validate

benchmark:
	python scripts/benchmark_hardware.py

report:
	python -m insilico_trial.cli report

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
