.PHONY: install lint typecheck test test-cov clean help demo demo-small demo-standard14 validate benchmark compare-organ-networks report check formal-gate formal-spec mutation

help:
	@echo "Available targets:"
	@echo "  install      - Install package with dev dependencies"
	@echo "  lint         - Run ruff linter"
	@echo "  typecheck    - Run mypy type checker"
	@echo "  test         - Run pytest test suite"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  demo-small   - Run 100-patient fast validation demo"
	@echo "  demo         - Run 1000-patient full SAD/MAD simulation"
	@echo "  demo-standard14 - Run the same 1000-patient SAD benchmark on the 14-organ network"
	@echo "  validate     - Run benchmark harness (Warfarin + Moxifloxacin + Midazolam + Metformin)"
	@echo "  formal-gate  - Run QED formal verification gate (portable)"
	@echo "  formal-spec  - Check formal_specs/*.tex against the PBPK model, prove each theorem, write output/validation/traces.json"
	@echo "  mutation     - Mutation kill-rate gate over the baseline targets"
	@echo "  benchmark    - Hardware acceleration comparison"
	@echo "  compare-organ-networks - DEFAULT vs STANDARD_14 Cmax delta + throughput"
	@echo "  report       - Generate HTML/Markdown validation report"
	@echo "  clean        - Remove build artifacts"

install:
	uv pip install --system -e ".[dev]"

lint:
	ruff check src/insilico_trial

typecheck:
	mypy src/insilico_trial

test:
	pytest src/insilico_trial/tests/ -x -q

test-cov:
	pytest src/insilico_trial/tests/ -x -q --cov=insilico_trial --cov-report=term-missing

demo-small:
	python -m insilico_trial.cli demo-small --patients 100 --duration-days 7

demo:
	python -m insilico_trial.cli demo --patients 1000 --duration-days 7

demo-standard14:
	python -m insilico_trial.cli demo --patients 1000 --duration-days 7 --organ-network STANDARD_14

compare-organ-networks:
	python3 scripts/compare_organ_networks.py --patients 1000 --duration-days 7

validate:
	python -m insilico_trial.cli validate --warfarin-n 300
	python3 scripts/export_pbpk_to_qed.py --out /tmp/pbpk_lemmas.txt --parametric
	python3 scripts/verify_formal_gate.py /tmp/pbpk_lemmas.txt --fin-n 6 --strict

mutation:
	python3 scripts/mutation_gate.py --json mutation_evidence.json

formal-gate:
	python3 scripts/export_pbpk_to_qed.py --out /tmp/pbpk_lemmas.txt --parametric
	python3 scripts/verify_formal_gate.py /tmp/pbpk_lemmas.txt --fin-n 6 --strict

formal-spec:
	python3 scripts/verify_formal_spec.py

benchmark:
	python -m insilico_trial.cli benchmark

report:
	python -m insilico_trial.cli report

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
