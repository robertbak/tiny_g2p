# tiny_g2p -- tiny parallel G2P for Polish, following gpu-lexer's experiments
#
# A per-character classifier (~15K params) trained on DP-aligned MFA labels,
# under gpu-lexer's regime: warm-start continuation, QAT, failure bank,
# fixed verification, guarded promotion.

SHELL := /bin/bash
PY    := uv run

# Where the Rust half is checked out. It trains here and runs there, so `export`
# writes the weights into that checkout and nothing else crosses.
RUST_DIR ?= ../tiny-g2p

.DEFAULT_GOAL := help

.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: sync lexicon ## install the env and make the lexicon available

.PHONY: sync
sync: ## create/refresh the uv virtualenv (python 3.12 + torch)
	uv sync --python 3.12

.PHONY: lexicon
lexicon: ## fetch the pinned MFA dictionary (or reuse the sibling copy)
	$(PY) tiny-g2p fetch

.PHONY: stats
stats: ## lexicon summary, alignment coverage and split sizes
	$(PY) tiny-g2p stats --splits

.PHONY: probe
probe: ## align the lexicon and report length/coverage stats (decides the design)
	$(PY) python experiments/probe_lexicon.py

.PHONY: train
train: ## train the model (EPOCHS=32, warm-starts the promoted checkpoint)
	$(PY) tiny-g2p train --epochs $(or $(EPOCHS),32)

.PHONY: eval
eval: ## score the promoted checkpoint on the held-out test words
	$(PY) tiny-g2p eval --show-errors 15

.PHONY: fine-tune
fine-tune: ## repair failure words: make fine-tune W="słowo inne"
	$(PY) tiny-g2p fine-tune $(W)

.PHONY: promote
promote: ## promote a run if it strictly improves under guards: make promote RUN=runs/<ts>
	$(PY) tiny-g2p promote $(RUN)

.PHONY: fetch-cv
fetch-cv: ## fetch the pinned Common Voice dictionary (the second opinion)
	$(PY) tiny-g2p fetch-cv

.PHONY: xlex
xlex: fetch-cv ## gold trust: MFA<->CV agreement and the model slices
	$(PY) python experiments/xlex_agreement.py --json runs/xlex_agreement.json
	$(PY) python experiments/xlex_rescore.py --json runs/xlex_rescore.json
	$(PY) python experiments/build_adjudication.py

.PHONY: adjudicate
adjudicate: ## measured vs gold-corrected accuracy (needs make xlex)
	$(PY) python experiments/score_adjudication.py --full --json runs/adjudication.json

.PHONY: export
export: ## package the promoted model into the Rust checkout (RUST_DIR)
	$(PY) tiny-g2p export \
		--blob $(RUST_DIR)/core/weights/tiny_g2p.bin \
		--gold data/test_gold.tsv \
		--onnx $(RUST_DIR)/core/weights/tiny_g2p.onnx \
		$(foreach a,agd bmw cv dga mps nszz ntv pzpr rpo rtv tpn wku wtw,--acronym $(a))
	@echo "wrote $(RUST_DIR)/core/weights/"; \
	 echo "build and test it there: make -C $(RUST_DIR) test"

.PHONY: test
test: ## run the test suite
	$(PY) pytest -q

.PHONY: predict
predict: ## transcribe words: make predict W="Wrocław Szczebrzeszyn"
	$(PY) tiny-g2p predict $(W)

.PHONY: clean
clean: ## remove the venv, runs, promoted state and downloaded lexicon
	rm -rf .venv runs active data/lexicons failure_bank .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
