#!/usr/bin/env bash
# Create an isolated pinned VeRL checkout for the training scripts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VERL_DIR="${VERL_DIR:-$HERE/.verl}"
REVISION="$(cat "$HERE/verl.pin")"

if [[ ! -d "$VERL_DIR/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
fi
git -C "$VERL_DIR" fetch origin "$REVISION"
git -C "$VERL_DIR" checkout --detach "$REVISION"
echo "Pinned VeRL checkout at $(git -C "$VERL_DIR" rev-parse HEAD)"
echo "Install VeRL's documented CUDA + SGLang + FSDP dependencies for this GPU image, then run:"
echo "  cd '$VERL_DIR' && python3 -m pip install -e ."
echo "The trainer also needs pandas, pyarrow, huggingface_hub, requests, transformers, and a compatible SGLang build."
