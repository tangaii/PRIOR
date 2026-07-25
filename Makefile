PYTHON ?= python
ARTIFACT_DIR ?= artifacts/frozen
OUTPUT ?= reproduction/prior_submission.jsonl

.PHONY: test compile reproduce reproduce-with-test validate neural-smoke

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

compile:
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests

reproduce:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_submission.py \
		--artifact-dir $(ARTIFACT_DIR) \
		--output $(OUTPUT)

reproduce-with-test:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_submission.py \
		--artifact-dir $(ARTIFACT_DIR) \
		--output $(OUTPUT) \
		--test-jsonl data/test.jsonl

validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_submission.py \
		--submission $(OUTPUT) \
		--test-jsonl data/test.jsonl \
		--expected-rows 638204

neural-smoke:
	PYTHONPATH=src $(PYTHON) scripts/infer.py \
		--input-jsonl tests/fixtures/inference_sample.jsonl \
		--model-dir artifacts/models/base_model \
		--checkpoint artifacts/models/anchor_expert.pt \
		--output-jsonl reproduction/neural_smoke_predictions.jsonl \
		--mode anchor \
		--batch-size 1
