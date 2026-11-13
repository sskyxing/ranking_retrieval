#!/usr/bin/env bash
set -euo pipefail

# Recreate the legacy 2020 runtime. Run in Linux/WSL with Python 3.8.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.8}
ENV_DIR=${ENV_DIR:-"$ROOT/paper-env"}

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 8):
    raise SystemExit(f"Python 3.8 is required, got {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade "pip==23.3.2" "setuptools==68.2.2" "wheel==0.41.3"

# Versions are deliberately bounded for the legacy Transformers imports used
# by official/. FAISS 1.6.x is preferred when the platform still provides it;
# 1.7.2 is the oldest broadly available fallback and must be recorded as a
# reproduction deviation.
python -m pip install \
  "torch==1.7.1" \
  "transformers==2.11.0" \
  "tokenizers==0.7.0" \
  "faiss-cpu==1.7.2" \
  "numpy==1.19.5" \
  "spacy==2.3.9" \
  "cython==0.29.36" \
  "filelock==3.12.2" \
  "regex==2023.10.3" \
  "tqdm==4.66.1" \
  "wget==3.2"
python -m pip install --no-deps -e "$ROOT/official"

python - <<'PY'
import faiss, numpy, torch, transformers
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "faiss": getattr(faiss, "__version__", "unknown"),
    "cuda": torch.cuda.is_available(),
})
PY
