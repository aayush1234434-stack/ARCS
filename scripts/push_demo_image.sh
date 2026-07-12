#!/usr/bin/env bash
# Build a cloud image with the ONNX router baked in, then push to a registry.
# Railway cannot upload model.onnx via `railway up` (per-file ~250MB Cloudflare limit).
#
# Usage:
#   ./scripts/push_demo_image.sh YOUR_DOCKERHUB_USER/arcs-demo:onnx
#   # then in Railway: Settings → Source → Docker Image → that tag

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${1:-}"
if [[ -z "$IMAGE" ]]; then
  echo "Usage: $0 <registry/image:tag>"
  echo "Example: $0 youruser/arcs-demo:onnx"
  exit 1
fi

if [[ ! -f artifacts/router-model/model.onnx ]]; then
  echo "Missing artifacts/router-model/model.onnx"
  echo "Run: python scripts/export_router_onnx.py"
  exit 1
fi

if [[ ! -f artifacts/router-model/config.json ]]; then
  echo "Missing artifacts/router-model/config.json"
  exit 1
fi

echo "Building $IMAGE (ONNX router baked; safetensors excluded by .dockerignore)…"
docker build -t "$IMAGE" .

echo "Pushing $IMAGE…"
docker push "$IMAGE"

cat <<EOF

Pushed: $IMAGE

Railway click path:
  1. Open your ARCS service
  2. Settings → Source / Network (or New Service → Docker Image)
  3. Image: $IMAGE
  4. Variables: GROQ_API_KEY, NVIDIA_API_KEY, ARCS_ROUTER_BACKEND=onnx, ARCS_DEMO_PUBLIC=1
  5. Settings → Networking → Generate Domain
  6. curl https://YOUR_DOMAIN/health

Do NOT use: railway up --no-gitignore  (model.onnx exceeds Railway upload limits)
EOF
