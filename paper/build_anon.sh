#!/usr/bin/env bash
# Builds the double-blind submission PDF (NeSyDebates @ KSE 2026 is double-blind).
# Flips the \anonfalse switch to \anontrue in a scratch copy, so main.tex stays
# the camera-ready source. Output: main_anon.pdf
set -euo pipefail
cd "$(dirname "$0")"
sed 's/^\\anonfalse$/\\anontrue/' main.tex > main_anon.tex
pdflatex -interaction=nonstopmode -halt-on-error main_anon.tex >/dev/null
bibtex main_anon >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main_anon.tex >/dev/null
pdflatex -interaction=nonstopmode main_anon.tex >/dev/null
echo "built main_anon.pdf"
