#!/usr/bin/env bash
set -euo pipefail

TAG_DATE="$(date +%Y%m%d)"
IMAGE="aykutmursalo/aykutci-${TAG_DATE}-fp8bf16"

echo "🔨  Building ${IMAGE}"
docker buildx build \
  --platform linux/amd64 \
  --tag "${IMAGE}" \
  --build-arg HF_TOKEN="${HF_TOKEN:-}" \
  --build-arg CIVI_TOKEN="${CIVI_TOKEN:-}" \
  --compress --squash --provenance=false \
  --push .

echo "✅  Pushed to Docker Hub: ${IMAGE}"
