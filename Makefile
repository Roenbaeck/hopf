PAPER := hopf_seam
PYTHON ?= python3
LATEXMK ?= latexmk
OUTPUT_DIR := output/pdf
BUILD_DIR := tmp/latex

.PHONY: all verify pdf clean

all: verify pdf

verify:
	$(PYTHON) verify_hessian_tori.py
	$(PYTHON) verify_mixed_curvature_counterexample.py
	$(PYTHON) verify_coupled_kernel_classification.py
	$(PYTHON) verify_coupled_kernel_obstruction.py
	$(PYTHON) verify_killing_kernel.py

pdf:
	mkdir -p $(OUTPUT_DIR) $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(OUTPUT_DIR)/$(PAPER).pdf

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(PAPER).tex
