#!/usr/bin/env bash
# Verify paper/main.tex compiles, using SYNTHETIC macro values in a temp dir.
#
# Never writes into paper/ -- no fabricated number can reach the repo. The synthetic
# macro list is derived from src/make_tables.py so it cannot drift from the generator.
# Also fails if the paper references a generated macro the generator does not emit.
#
# Usage: bash src/check_build.sh
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(mktemp -d)"
trap 'rm -rf "$WS"' EXIT

grep -oE 'mac\("[A-Za-z]+"' "$REPO/src/make_tables.py" | sed 's/mac("//;s/"//' | sort -u > "$WS/emitted"
grep -oE '\\[A-Za-z]+' "$REPO/paper/main.tex" | sed 's/^\\//' | sort -u > "$WS/used"

# generated-macro names are CamelCase; ignore LaTeX/IEEEtran built-ins by allowlist
BUILTIN="Delta IEEEauthorblockN IEEEauthorblockA IEEEoverridecommandlockouts IEEEtran"
MISSING=""
for m in $(comm -13 "$WS/emitted" <(grep -E '^[A-Z]' "$WS/used")); do
  case " $BUILTIN " in *" $m "*) ;; *) MISSING="$MISSING $m";; esac
done
if [ -n "$MISSING" ]; then
  echo "FAIL: paper uses macros the generator does not emit:$MISSING" >&2; exit 1
fi

cp "$REPO/paper/main.tex" "$REPO/paper/refs.bib" "$WS/"
: > "$WS/macros.tex"
while read -r m; do
  printf '\\newcommand{\\%s}{0}\n' "$m" >> "$WS/macros.tex"
done < "$WS/emitted"
printf '\\begin{tabular}{lrr}\\hline x & 0 & 0 \\\\ \\hline\\end{tabular}\n' > "$WS/table_main.tex"

cd "$WS"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > b1.log 2>&1 || { tail -25 b1.log; exit 1; }
bibtex main > bib.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > b2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > b3.log 2>&1 || true

[ -f main.pdf ] || { echo "FAIL: no PDF produced" >&2; exit 1; }
echo "pages:      $(grep -oE 'Output written on main\.pdf \([0-9]+ page' b3.log | grep -oE '[0-9]+' || echo '?')"
echo "macros:     $(wc -l < "$WS/emitted" | tr -d ' ') synthetic values injected"
UNDEF=$(grep -c "undefined" b3.log || true)
echo "undefined:  $UNDEF"
[ "$UNDEF" -gt 0 ] && grep -i "undefined" b3.log | head -5
echo "bibtex:     $(grep -ciE 'warning|error' bib.log || true) warnings/errors"
echo "overfull:   $(grep -c Overfull b3.log || true) boxes"
[ "$UNDEF" -eq 0 ] && echo "PASS: clean build, no undefined citations or references"
