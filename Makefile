.PHONY: all experiments paper clean

all: paper

experiments:
	python experiments/run_strong_experiments.py

experiments-lite:
	python experiments/run_all.py

paper:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
