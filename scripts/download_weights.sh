#!/usr/bin/env bash
# Download the GroundingDINO and MobileSAM weights into weights/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
curl -fL -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl -fL -o weights/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
