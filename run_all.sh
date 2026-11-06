#!/usr/bin/env bash
# Reproduce the four-model refusal study, then regenerate the paper assets.
#   ./run_all.sh          full run
#   ./run_all.sh --quick  smoke test
set -euo pipefail
cd "$(dirname "$0")"

PY="./venv/bin/python"
[ -x "$PY" ] || PY="python3"

EXTRA="${1:-}"
MODELS=(qwen35_08b r1distill_15b lfm3_17b qwen35_4b)

for M in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "MODEL: ${M}"
    echo "======================================================================"
    case "$M" in
        r1distill_15b) LOCAL="models/r1" ;;
        qwen35_4b)     LOCAL="models/qwen35_4b" ;;
        *)             LOCAL="" ;;
    esac
    if [ -n "$LOCAL" ] && [ -d "$LOCAL" ]; then
        "$PY" src/study.py --model "$M" --model-path "$LOCAL" ${EXTRA}
    else
        "$PY" src/study.py --model "$M" ${EXTRA}
    fi
done

"$PY" src/make_tables.py
"$PY" src/make_figures.py

echo ""
echo "Done. See paper/main.tex and results/summary_all.csv"
