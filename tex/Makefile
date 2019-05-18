.PHONY: run

run:
	pdflatex article.tex && pdflatex article.tex && tput reset
	# pdflatex article.tex && bibtex article && pdflatex article.tex && tput reset

.PHONY: clean

clean:
	rm -f *.aux *.lof *.log *.lot *.out *.toc *.pdf *.ist *.blg *.bbl
