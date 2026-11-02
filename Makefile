.PHONY: install download test smoke run-all controls report mutation docker docker-test

PY ?= $(firstword $(wildcard .venv/Scripts/python.exe .venv/bin/python))
IMAGE ?= eeg-moabb

install:
	[ -d .venv ] || uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]"

download:
	bash scripts/download.sh

# Offline and CPU only. Tests that need the dataset skip when it is not downloaded.
test:
	$(PY) -m pytest --junitxml=results/test_results.xml

# 2 subjects, both pipelines, both protocols, then the label-shuffle control on the same subjects.
smoke:
	bash scripts/run_all.sh --smoke
	bash scripts/run_controls.sh --smoke

run-all:
	bash scripts/run_all.sh

controls:
	bash scripts/run_controls.sh

# FINDINGS.md, README.md and CLAIMS.md, rendered from results/. Fails on an unverified claim.
report:
	$(PY) -m eeg.report

# Re-runs the leakage tests with the audit disabled; the leak tests must then fail.
mutation:
	$(PY) scripts/mutation_check.py

docker:
	docker build -t $(IMAGE) .
	mkdir -p results
	docker image inspect $(IMAGE) --format '{{.Id}}' > results/docker_build.txt

# The data-free tests, inside the image (tests that need the dataset skip there).
docker-test:
	docker run --rm $(IMAGE) python -m pytest -q -p no:cacheprovider
