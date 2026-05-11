RELATORIO_DIR := docs/relatorio
RELATORIO_TEX := relatorio.tex
LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: relatorio clean-relatorio clean

relatorio:
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"
	rm -f "$(RELATORIO_DIR)"/*.aux
	rm -f "$(RELATORIO_DIR)"/*.bbl
	rm -f "$(RELATORIO_DIR)"/*.bcf
	rm -f "$(RELATORIO_DIR)"/*.blg
	rm -f "$(RELATORIO_DIR)"/*.fdb_latexmk
	rm -f "$(RELATORIO_DIR)"/*.fls
	rm -f "$(RELATORIO_DIR)"/*.log
	rm -f "$(RELATORIO_DIR)"/*.out
	rm -f "$(RELATORIO_DIR)"/*.run.xml
	rm -f "$(RELATORIO_DIR)"/*.synctex.gz
	rm -f "$(RELATORIO_DIR)"/*.toc
	