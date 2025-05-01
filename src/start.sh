#!/usr/bin/env bash
set -e

###############################################################################
# 0)   KULLANICI AYARLARI
###############################################################################
VOL_ROOT="${VOL_ROOT:-/runpod-volume}"       # network volume mount path
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_HOST="127.0.0.1"

###############################################################################
# 1)   MODEL KONTROLÜ  (HiDream + Flux)
###############################################################################
HIDREAM=/comfyui/models/diffusion_models/hidream_i1_dev_bf16.safetensors
FLUX=/comfyui/models/diffusion_models/fluxFillFP8_v10.safetensors

echo "🔍  Checking model files..."
for f in "$HIDREAM" "$FLUX"; do
  [[ -f "$f" ]] || { echo "❌  Model file missing: $f"; exit 1; }
done
echo "✅  All required models are present."

###############################################################################
# 2)   VOLUME ISITMA  (ilk pod’da kopyala)
###############################################################################
if [ ! -f "${VOL_ROOT}/.ready" ]; then
  echo "🟡  Network volume boş; modeller ve nodelar kopyalanıyor…"
  mkdir -p "${VOL_ROOT}"
  rsync -a --ignore-existing /comfyui/models/   "${VOL_ROOT}/models/"
  rsync -a --ignore-existing /comfyui/custom_nodes/ "${VOL_ROOT}/custom_nodes/"
  touch "${VOL_ROOT}/.ready"
  echo "🟢  Kopyalama tamamlandı."
fi

###############################################################################
# 3)   BELLEK OPTİMİZASYONU
###############################################################################
TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d+' | head -n1 || true)"
[[ -n "$TCMALLOC" ]] && export LD_PRELOAD="$TCMALLOC"

###############################################################################
# 4)   COMFYUI + RUNPOD HANDLER
###############################################################################
echo "🚀  Starting ComfyUI (API-only)…"
python3 /comfyui/main.py \
        --listen "${COMFY_HOST}" \
        --port "${COMFY_PORT}" \
        --disable-auto-launch \
        --disable-metadata &

echo "✅  ComfyUI listening on ${COMFY_HOST}:${COMFY_PORT}"

if [[ "$SERVE_API_LOCALLY" == "true" ]]; then
  echo "🔌  Starting RunPod handler (dev mode)…"
  python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  python3 -u /rp_handler.py
fi
