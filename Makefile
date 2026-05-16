RELATORIO_DIR := docs/relatorio
RELATORIO_TEX := relatorio.tex
LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error
CLEAN_RELATORIO_FILES := *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc

.PHONY: all relatorio clean-relatorio clean

all: relatorio

relatorio:
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"
	cd "$(RELATORIO_DIR)" && $(LATEX) $(LATEX_FLAGS) "$(RELATORIO_TEX)"
	$(MAKE) clean-relatorio

clean-relatorio:
	rm -f $(addprefix "$(RELATORIO_DIR)"/,$(CLEAN_RELATORIO_FILES))

clean:
	$(MAKE) clean-relatorio
