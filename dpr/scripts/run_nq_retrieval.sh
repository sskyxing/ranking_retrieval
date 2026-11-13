#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OFFICIAL="$ROOT/official"
DOWNLOADS=${DOWNLOADS:-"$ROOT/downloads"}
OUTPUT=${OUTPUT:-"$ROOT/outputs/nq-test.json"}
MODEL=${MODEL:-"$DOWNLOADS/checkpoint/retriever/single/nq/bert-base-encoder.cp"}
WIKI=${WIKI:-"$DOWNLOADS/data/wikipedia_split/psgs_w100.tsv"}
QA=${QA:-"$DOWNLOADS/data/retriever/qas/nq-test.csv"}
EMBEDDINGS=${EMBEDDINGS:-"$DOWNLOADS/data/retriever_results/nq/single/wikipedia_passages_*.pkl"}

for path in "$MODEL" "$WIKI" "$QA"; do
  [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 2; }
done
compgen -G "$EMBEDDINGS" >/dev/null || { echo "No embeddings match $EMBEDDINGS" >&2; exit 2; }
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("The paper-era dense_retriever.py requires CUDA for query encoding")
PY
if [[ -r /proc/meminfo ]]; then
  AVAILABLE_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( AVAILABLE_KIB < 100 * 1024 * 1024 )); then
    echo "At least about 100 GiB free RAM is required; found $((AVAILABLE_KIB / 1024 / 1024)) GiB." >&2
    exit 2
  fi
fi
mkdir -p "$(dirname "$OUTPUT")"
cd "$OFFICIAL"
python dense_retriever.py \
  --model_file "$MODEL" \
  --ctx_file "$WIKI" \
  --qa_file "$QA" \
  --encoded_ctx_file "$EMBEDDINGS" \
  --out_file "$OUTPUT" \
  --n-docs 100 \
  --validation_workers 32 \
  --batch_size 64
