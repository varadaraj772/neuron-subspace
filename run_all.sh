#!/usr/bin/env bash
# Reproduce the four-model refusal study.
#   ./run_all.sh          full run
#   ./run_all.sh --quick  smoke test
set -euo pipefail
cd "$(dirname "$0")"

if [ -d venv ]; then
    source venv/bin/activate
fi

EXTRA="${1:-}"
MODELS=(qwen35_08b r1distill_15b lfm3_17b qwen35_4b)

for M in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "MODEL: ${M}"
    echo "======================================================================"
    python src/study.py --model "${M}" ${EXTRA}
done

python src/make_report.py
echo ""
echo "Done. See FINAL_REPORT.md and results/summary_all.csv"
