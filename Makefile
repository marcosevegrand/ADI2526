RELATORIO_DIR := docs/relatorio
RELATORIO_TEX := relatorio.tex
LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all relatorio clean-relatorio clean

all: relatorio

relatorio:
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"

clean:
	rm -f "$(RELATORIO_DIR)"/*.aux "$(RELATORIO_DIR)"/*.bbl "$(RELATORIO_DIR)"/*.bcf "$(RELATORIO_DIR)"/*.blg "$(RELATORIO_DIR)"/*.fdb_latexmk "$(RELATORIO_DIR)"/*.fls "$(RELATORIO_DIR)"/*.log "$(RELATORIO_DIR)"/*.out "$(RELATORIO_DIR)"/*.run.xml "$(RELATORIO_DIR)"/*.synctex.gz "$(RELATORIO_DIR)"/*.toc

