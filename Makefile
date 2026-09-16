.PHONY: install test smoke demo report build

install:
	python -m pip install -e ".[dev,optimization]"

test:
	pytest -q

smoke:
	python scripts/smoke_e2e.py --dry-run --json

demo:
	python scripts/run_demo.py

report:
	@test -n "$(EXPERIMENT)" || (echo "usage: make report EXPERIMENT=artifacts/experiments/<run>"; exit 2)
	python -m arcs.eval.report "$(EXPERIMENT)"

build:
	python -m build
