#!/usr/bin/env bash
# Builds the version permitted for arXiv / a personal site: the accepted paper
# with the IEEE copyright notice on page 1. IEEE policy allows the accepted
# article on the author's personal site, an institutional repository, arXiv.org
# or TechRxiv.org, and requires that notice. Output: main_postprint.pdf
set -euo pipefail
cd "$(dirname "$0")"
sed 's/^\\postprintfalse$/\\postprinttrue/' main.tex > main_postprint.tex
pdflatex -interaction=nonstopmode -halt-on-error main_postprint.tex >/dev/null
bibtex main_postprint >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main_postprint.tex >/dev/null
pdflatex -interaction=nonstopmode main_postprint.tex >/dev/null
echo "built main_postprint.pdf"
