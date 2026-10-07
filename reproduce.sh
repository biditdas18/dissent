#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from the committed run
# logs. No API calls, no network, no model access required.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] frozen sample (re-downloads raw benchmarks on first run)"
python src/build_sample.py

echo "[2/4] analysis from committed run logs"
python src/analyze.py

echo "[3/4] tables, macros and figures"
python src/make_tables.py
python src/make_figures.py

echo "[4/4] manuscript"
# Steps 1-3 are the reproducibility claim and need no LaTeX. Step 4 is a
# convenience rebuild. It used to print "done" even when pdflatex was absent,
# which reported a success that had not happened, so it now says which it did.
export PATH="/Library/TeX/texbin:$PATH"
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "  SKIPPED: pdflatex not found. All numbers, tables and figures above are"
  echo "  reproduced; only the PDF rebuild needs a LaTeX install."
  echo "done -> analysis/, paper/macros.tex, paper/table_*.tex, figures/ (PDF not rebuilt)"
  exit 0
fi
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "done -> paper/main.pdf"
