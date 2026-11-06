PY ?= python
STEPS ?=

.PHONY: help install assets all english telugu preprocess train infer plots report serve docker test smoke lint clean-smoke

help:
	@echo "make install      install Python dependencies"
	@echo "make assets       download HiFi-GAN vocoder + NLTK data"
	@echo "make all          full pipeline for both languages (needs HF_TOKEN for Svarah)"
	@echo "make english      full pipeline, English only   (make telugu likewise)"
	@echo "make report       rebuild Report/Technical_Report.pdf"
	@echo "make serve        run the web app on http://localhost:7860"
	@echo "make docker       build and run the Docker image"
	@echo "make test         run unit + end-to-end smoke tests"

install:
	$(PY) -m pip install -r requirements.txt

assets:
	$(PY) scripts/download_assets.py --vocoder --nltk

all:
	STEPS=$(STEPS) bash scripts/run_all.sh english telugu

english:
	STEPS=$(STEPS) bash scripts/run_all.sh english

telugu:
	STEPS=$(STEPS) bash scripts/run_all.sh telugu

preprocess:
	$(PY) English/preprocess.py && $(PY) Telugu/preprocess.py

train:
	$(PY) English/train.py && $(PY) Telugu/train.py

plots:
	$(PY) English/train.py --plot-only && $(PY) Telugu/train.py --plot-only

infer:
	$(PY) English/inference.py && $(PY) Telugu/inference.py

report:
	$(PY) -m fastspeech2.cli report --config English/config.yaml

serve:
	$(PY) -m uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

docker:
	docker compose up --build

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m pytest -q tests/test_pipeline.py

clean-smoke:
	rm -rf data/smoke
