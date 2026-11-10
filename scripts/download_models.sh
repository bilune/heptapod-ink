#!/usr/bin/env bash
# Download the published Heptapod Ink weights from the Hugging Face model repo.
#
#   HF_REPO=<user>/heptapod-ink bash scripts/download_models.sh
#
# Model repo layout (built by hf/publish.sh):
#   model.pt                -> v4/runs/s2/model.pt   (PyTorch checkpoint, best v4 run)
#   web/<file>              -> web/models/<file>     (ONNX exports for the browser demo)
#
# Uses the `hf` CLI (pip install huggingface_hub) when available, plain curl otherwise.
# Set HF_REVISION to pin a commit or tag (default: main). Set ONLY=pt or ONLY=web to fetch one part; ONLY=data fetches the training splits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="${HF_REPO:-<user>/heptapod-ink}"
REV="${HF_REVISION:-main}"
ONLY="${ONLY:-}"

if [[ "$HF_REPO" == *"<user>"* ]]; then
  echo "Set HF_REPO to the model repo id, e.g. HF_REPO=someone/heptapod-ink bash scripts/download_models.sh" >&2
  exit 1
fi

WEB_FILES=(
  config.json
  decoder_fp16.onnx
  eyes_q8pc.onnx
  memory.onnx
  render.onnx
  sym_memory.onnx
  writer_alt_mix.onnx
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
)

HF_BIN=""
if command -v hf >/dev/null 2>&1; then HF_BIN="hf"
elif [ -x "$ROOT/.venv/bin/hf" ]; then HF_BIN="$ROOT/.venv/bin/hf"
fi

fetch() {  # fetch <path in repo> <local destination>
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [ -n "$HF_BIN" ]; then
    local stage="$ROOT/.cache/hf-download"
    "$HF_BIN" download "$HF_REPO" "$src" --revision "$REV" --local-dir "$stage" >/dev/null
    cp "$stage/$src" "$dst"
  else
    curl -fL --retry 3 -o "$dst" "https://huggingface.co/$HF_REPO/resolve/$REV/$src"
  fi
  echo "ok  $dst"
}

if [ "$ONLY" != "web" ] && [ "$ONLY" != "data" ]; then
  fetch model.pt "$ROOT/v4/runs/s2/model.pt"
fi
if [ "$ONLY" != "pt" ] && [ "$ONLY" != "data" ]; then
  for f in "${WEB_FILES[@]}"; do
    fetch "web/$f" "$ROOT/web/models/$f"
  done
fi
if [ "$ONLY" = "data" ]; then  # opt-in: exact training splits (~180 MB), only needed to retrain
  for f in train.json mixed_train.json mixed_val.json; do
    fetch "data/$f" "$ROOT/v3/data_tatoeba/$f"
  done
fi
rm -rf "$ROOT/.cache/hf-download"
echo "done: v4/runs/s2/model.pt and web/models/ are in place (python web/serve.py for the demo)"
