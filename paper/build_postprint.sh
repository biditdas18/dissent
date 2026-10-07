#!/usr/bin/env bash
# Builds the copy for public posting: the accepted paper with a provenance line
# on page 1 naming the venue and marking it the author's accepted version.
# NeSyDebates proceedings are a standalone volume, not IEEE-published and not
# IEEE-indexed, so no IEEE copyright notice applies. Output: main_postprint.pdf
set -euo pipefail
cd "$(dirname "$0")"
sed 's/^\\postprintfalse$/\\postprinttrue/' main.tex > main_postprint.tex
pdflatex -interaction=nonstopmode -halt-on-error main_postprint.tex >/dev/null
bibtex main_postprint >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main_postprint.tex >/dev/null
pdflatex -interaction=nonstopmode main_postprint.tex >/dev/null
echo "built main_postprint.pdf"
