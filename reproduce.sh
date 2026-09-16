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
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "done -> paper/main.pdf"
