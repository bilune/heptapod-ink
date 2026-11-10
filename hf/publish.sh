#!/usr/bin/env bash
# Assemble and upload the Hugging Face model repo and the static Space.
#   HF_USER=<your-hf-username> bash hf/publish.sh          # build + upload
#   HF_USER=<your-hf-username> DRY_RUN=1 bash hf/publish.sh # build only, into hf/build/
# Needs: `hf auth login` (the hf CLI from huggingface_hub), v4/runs/s2/model.pt and web/models/.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_USER:?set HF_USER to your Hugging Face username}"
NAME="${HF_NAME:-heptapod-ink}"
B=hf/build
rm -rf "$B" && mkdir -p "$B/model/web" "$B/space"

# model repo: checkpoint + ONNX exports + card
cp v4/runs/s2/model.pt "$B/model/model.pt"
cp -R web/models/. "$B/model/web/"
cp NOTICE.md "$B/model/NOTICE.md"
mkdir -p "$B/model/data" && cp v3/data_tatoeba/*.json "$B/model/data/"   # exact splits used for training and the exams
sed "s#<user>#$HF_USER#g" hf/model/README.md > "$B/model/README.md"

# space: the browser demo, models served from the same origin
cp web/index.html web/app.js "$B/space/"
cp -R web/models "$B/space/models"
sed "s#<user>#$HF_USER#g" hf/space/README.md > "$B/space/README.md"

du -sh "$B"/*
[ -n "${DRY_RUN:-}" ] && { echo "dry run: built $B, nothing uploaded"; exit 0; }

hf repos create "$HF_USER/$NAME" --type model --public --exist-ok
hf upload "$HF_USER/$NAME" "$B/model" . --repo-type model --commit-message "Upload Heptapod Ink weights"
hf repos create "$HF_USER/$NAME" --type space --sdk static --public --exist-ok
hf upload "$HF_USER/$NAME" "$B/space" . --repo-type space --commit-message "Upload Heptapod Ink demo"
echo "model: https://huggingface.co/$HF_USER/$NAME"
echo "space: https://huggingface.co/spaces/$HF_USER/$NAME"
