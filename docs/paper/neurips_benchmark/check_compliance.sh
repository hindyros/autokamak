#!/usr/bin/env bash
# Pre-submission compliance check.
#
#   ./check_compliance.sh            # draft mode: warn, don't fail
#   ./check_compliance.sh --strict   # submission mode: any FAIL exits non-zero
#
# This checks what is mechanically checkable. It does NOT replace reading the
# call for papers -- page limits and required sections change year to year.

set -uo pipefail
cd "$(dirname "$0")"

STRICT=0
[[ "${1:-}" == "--strict" ]] && STRICT=1

FAILED=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
head2() { printf '\n\033[1m%s\033[0m\n' "$1"; }

[[ -f main.pdf ]] || { echo "main.pdf not found -- run 'latexmk -pdf main.tex' first."; exit 2; }

# --------------------------------------------------------------- 1. build ---
head2 "1. Build integrity"

if [[ -f main.log ]]; then
  if grep -qE "Citation .* undefined|Reference .* undefined" main.log; then
    fail "undefined references or citations (see main.log)"
  else
    pass "no undefined references or citations"
  fi
  if grep -q "LaTeX Error" main.log; then
    fail "LaTeX errors in main.log"
  else
    pass "no LaTeX errors"
  fi
else
  warn "main.log absent -- cannot check build warnings"
fi

if command -v pdffonts >/dev/null 2>&1; then
  if pdffonts main.pdf | awk 'NR>2 && $2=="Type3"' | grep -q .; then
    fail "Type 3 bitmap fonts present -- most venues reject these"
  elif pdffonts main.pdf | awk 'NR>2 && $4=="no"' | grep -q .; then
    fail "not all fonts are embedded"
  else
    pass "all fonts embedded, no Type 3"
  fi
else
  warn "pdffonts unavailable -- skipping font embedding check"
fi

# ------------------------------------------------------- 2. open slots -----
head2 "2. Unresolved content"

N_FILL=$(grep -roE '\\FILL\{|\\begin\{FILLBOX\}' sections main.tex 2>/dev/null | wc -l | tr -d ' ')
N_TBD=$(grep -roE '\\tbd' sections 2>/dev/null | wc -l | tr -d ' ')

if [[ "$N_FILL" -gt 0 ]]; then
  fail "$N_FILL unresolved FILL slot(s) -- see the index on the last page"
else
  pass "no FILL slots remain"
fi
if [[ "$N_TBD" -gt 0 ]]; then
  fail "$N_TBD unfilled table cell(s) (\\tbd)"
else
  pass "no unfilled table cells"
fi
if grep -q '\\listoffills' main.tex; then
  warn "\\listoffills is still in main.tex -- remove it before submitting"
else
  pass "open-slots index removed"
fi

# --------------------------------------------------------- 3. page limit ---
head2 "3. Page limit"

# The venue excludes acknowledgments, references, the checklist and appendices
# from the limit, so the countable body ends at whichever of those comes first.
# A body that fills page 9 exactly puts that heading at the TOP of page 10,
# which is compliant -- so the page containing the heading only counts if
# countable text precedes it on that page.
TOTAL=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
ENDPAGE=""
for p in $(seq 1 "$TOTAL"); do
  PAGE=$(pdftotext -f "$p" -l "$p" main.pdf - 2>/dev/null)
  HEAD=$(printf '%s' "$PAGE" | grep -nE '^[[:space:]]*(References|Acknowledgments( and Disclosure of Funding)?)[[:space:]]*$' | head -1 | cut -d: -f1)
  if [[ -n "$HEAD" ]]; then
    BEFORE=$(printf '%s' "$PAGE" | sed -n "1,$((HEAD-1))p" | tr -d '[:space:]' | wc -c | tr -d ' ')
    if [[ "$BEFORE" -gt 40 ]]; then ENDPAGE=$p; else ENDPAGE=$((p-1)); fi
    break
  fi
done

if [[ -n "$ENDPAGE" ]]; then
  echo "        countable body ends on page $ENDPAGE of $TOTAL"
  if [[ "$ENDPAGE" -gt 9 ]]; then
    fail "body exceeds 9 pages -- CONFIRM the limit in the call for papers"
  else
    pass "body within 9 pages (acknowledgments, references, checklist and appendices are excluded)"
  fi
else
  warn "could not locate the References or Acknowledgments heading -- check by hand"
fi

# ------------------------------------------------------- 4. anonymity ------
head2 "4. Anonymity (only relevant for double-blind submission)"

if grep -qE '^\s*\\usepackage\[[^]]*\b(preprint|final)\b' main.tex; then
  warn "style is in preprint/final mode -- authors are NAMED and line numbers are off"
  echo "        for review use: \\usepackage[eandd]{neurips_2026}"
else
  pass "anonymous mode: author block blanked, line numbers on"
  LEAKS=$(grep -rniE 'hindyros|@mit\.edu|github\.com/[A-Za-z0-9_-]+|Massachusetts Institute' \
            sections main.tex 2>/dev/null | grep -v '^\s*%' || true)
  if [[ -n "$LEAKS" ]]; then
    fail "de-anonymising strings found in an anonymous submission:"
    echo "$LEAKS" | sed 's/^/        /'
  else
    pass "no obvious de-anonymising strings"
  fi
fi

# ------------------------------------------ 5. venue-required components ---
head2 "5. Required components"

# Match actual answered items (\answerYes{}), not prose mentioning the macro.
if grep -rqE '\\answer(Yes|No|NA)\{' sections 2>/dev/null; then
  pass "Paper Checklist present"
else
  fail "NeurIPS Paper Checklist MISSING -- mandatory, copy it from the official template"
fi

if grep -q 'ProvidesPackage{neurips_2026}\[2026/05/01 v1.0 NeurIPS-style layout\]' neurips_2026.sty 2>/dev/null; then
  fail "still using the reconstructed style file -- replace with the official .sty"
else
  pass "style file is not the local reconstruction"
fi

if grep -rq '% VERIFY' references.bib 2>/dev/null; then
  N=$(grep -c '% VERIFY' references.bib)
  warn "$N bibliography entry/entries still flagged '% VERIFY'"
else
  pass "no unverified bibliography entries"
fi

# ------------------------------------------------------------- summary -----
head2 "Summary"
if [[ "$FAILED" -eq 0 ]]; then
  echo "  No blocking issues found."
else
  echo "  Blocking issues above. Re-read the call for papers before submitting."
fi
[[ "$STRICT" -eq 1 ]] && exit "$FAILED"
exit 0
