#!/usr/bin/env bash
# Compile the paper. Requires a TeX distribution (TeX Live / MacTeX / BasicTeX).
set -euo pipefail
cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

echo "built paper/main.pdf"
