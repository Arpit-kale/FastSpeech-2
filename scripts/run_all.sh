#!/usr/bin/env bash
# Full automated pipeline for both languages:
#   assets -> download/select -> features -> train -> loss plots -> test synthesis + variance plots -> report
#
#   export HF_TOKEN=hf_xxx            # required for the gated Svarah dataset
#   bash scripts/run_all.sh           # both languages
#   bash scripts/run_all.sh telugu    # one language
#   STEPS=20000 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LANGS=${*:-"english telugu"}
EXTRA=()
if [[ -n "${STEPS:-}" ]]; then EXTRA+=(--set "train.total_steps=${STEPS}"); fi

echo "==> Downloading vocoder + NLTK data"
python scripts/download_assets.py --vocoder --nltk

for lang in $LANGS; do
  dir="$(tr '[:lower:]' '[:upper:]' <<< "${lang:0:1}")${lang:1}"
  if [[ "$lang" == "english" && -z "${HF_TOKEN:-}" && ! -f data/english/raw/manifest.jsonl ]]; then
    echo "!! HF_TOKEN is not set. Svarah is gated: accept its terms on huggingface.co and export HF_TOKEN." >&2
    exit 1
  fi
  echo "==> [$dir] prepare + preprocess"
  python "$dir/preprocess.py" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "==> [$dir] train (+ loss plots)"
  python "$dir/train.py" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "==> [$dir] synthesise test sentences (+ variance plots)"
  python "$dir/inference.py" ${EXTRA[@]+"${EXTRA[@]}"}
done

echo "==> Building Report/Technical_Report.pdf"
python -m fastspeech2.cli report --config English/config.yaml
echo "Done. Fill Report/observations.yaml with listening notes and re-run 'make report'."
