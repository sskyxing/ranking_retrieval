#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OFFICIAL="$ROOT/official"
DOWNLOADS=${DOWNLOADS:-"$ROOT/downloads"}
OUTPUT=${OUTPUT:-"$ROOT/runs/nq-paper"}
WORLD_SIZE=${WORLD_SIZE:-8}

TRAIN="$DOWNLOADS/data/retriever/nq-train.json"
DEV="$DOWNLOADS/data/retriever/nq-dev.json"
[[ -f "$TRAIN" ]] || { echo "Missing $TRAIN" >&2; exit 2; }
[[ -f "$DEV" ]] || { echo "Missing $DEV" >&2; exit 2; }
[[ "$WORLD_SIZE" -eq 8 ]] || echo "WARNING: paper effective batch 128 assumes 8 GPUs x 16." >&2
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the paper-scale training path")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
PY
mkdir -p "$OUTPUT"
cd "$OFFICIAL"
python -m torch.distributed.launch --nproc_per_node="$WORLD_SIZE" train_dense_encoder.py \
  --max_grad_norm 2.0 \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg bert-base-uncased \
  --seed 12345 \
  --sequence_length 256 \
  --warmup_steps 1237 \
  --batch_size 16 \
  --do_lower_case \
  --train_file "$TRAIN" \
  --dev_file "$DEV" \
  --output_dir "$OUTPUT" \
  --learning_rate 2e-5 \
  --num_train_epochs 40 \
  --dev_batch_size 16 \
  --hard_negatives 1 \
  --other_negatives 0 \
  --val_av_rank_start_epoch 30
