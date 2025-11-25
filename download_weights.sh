#!/usr/bin/env bash
set -e

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

BASE_URL="https://huggingface.co/hantian/yolo-doclaynet/resolve/main"

echo "🔽 Downloading YOLO-DocLayNet weights from hantian/yolo-doclaynet ..."

# Nano model (fastest, best for CPU)
echo "→ yolov10n-doclaynet.pt"
wget -O "$WEIGHTS_DIR/yolov10n-doclaynet.pt" \
  "${BASE_URL}/yolov10n-doclaynet.pt?download=1"

# Small model (still CPU-friendly, more accurate)
echo "→ yolov10s-doclaynet.pt"
wget -O "$WEIGHTS_DIR/yolov10s-doclaynet.pt" \
  "${BASE_URL}/yolov10s-doclaynet.pt?download=1"

# (Optional) Medium model – uncomment if you want it
echo "→ yolov10m-doclaynet.pt"
wget -O "$WEIGHTS_DIR/yolov10m-doclaynet.pt" \
  "${BASE_URL}/yolov10m-doclaynet.pt?download=1"

echo
echo "✅ Download complete!"
echo "Weights saved in: $WEIGHTS_DIR"
echo
echo "Recommended default for CPU:"
echo "  model_path: 'weights/yolov10n-doclaynet.pt'   # nano (fastest)"
echo "or:"
echo "  model_path: 'weights/yolov10s-doclaynet.pt'   # small (better accuracy)"
