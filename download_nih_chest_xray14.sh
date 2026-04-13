#!/usr/bin/env bash
set -euo pipefail

uv run prepare-nih-chest-xray14 \
  --dataset-id BahaaEldin0/NIH-Chest-Xray-14 \
  --data-root data \
  "$@"
