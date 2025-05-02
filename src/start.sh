#!/usr/bin/env bash
set -e

###############################################################################
# 0)   KULLANICI AYARLARI - YÜKSEK PERFORMANS MODU
###############################################################################
VOL_ROOT="${VOL_ROOT:-/runpod-volume}"       # network volume mount path
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"

# Yüksek performans sistemini tanımla
export HIGH_MEMORY_SYSTEM=1
export HIGH_VRAM_SYSTEM=1

echo "🚀 Yüksek Performans Modu (80GB VRAM + 200GB RAM) etkin"

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
# 2)   YÜKSEK PERFORMANS OPTIMIZASYONLARI
###############################################################################
# tcmalloc optimizasyonları - aşırı büyük bellek için
TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d+' | head -n1 || true)"
if [[ -n "$TCMALLOC" ]]; then
  export LD_PRELOAD="$TCMALLOC"
  # Büyük bellek raporlama sınırını artır (50GB)
  export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=53687091200
  # Bellek sızıntılarını önlemek için daha agresif ayar
  export TCMALLOC_RELEASE_RATE=100.0
  # Büyük sayfa desteği (huge pages)
  export TCMALLOC_HEAP_LIMIT_MB=204800  # ~200GB
fi

# CUDA bellek optimizasyonları - büyük VRAM için özel ayarlar
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:2048,garbage_collection_threshold:0.9"

# NVIDIA sürücü optimizasyonları
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_SIZE=1073741824  # 1GB shader cache
export __GL_THREADED_OPTIMIZATIONS=1
export CUDA_CACHE_SIZE=1073741824  # 1GB CUDA cache
export CUDA_AUTO_BOOST=1
export CUDA_MODULE_LOADING=LAZY

# Python optimizasyonları
export PYTHONHASHSEED=1
export PYTHONUNBUFFERED=1
# GC optimizasyonları
export PYDEVD_DISABLE_FILE_VALIDATION=1
export PYTHONFAULTHANDLER=1

# CPU thread optimizasyonları
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

###############################################################################
# 3)   VOLUME ISITMA - PARALEL KOPYALAMA (YÜKSEK PERFORMANS)
###############################################################################
if [ ! -f "${VOL_ROOT}/.ready" ]; then
  echo "🟡  Network volume hazırlanıyor (paralel kopyalama)..."
  mkdir -p "${VOL_ROOT}/models" "${VOL_ROOT}/custom_nodes"
  
  # IO performansını artırmak için önbelleği temizle
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  
  # Paralel kopyalama işlemi (büyük dosya transfer performansı için)
  echo "📂 Modelleri ve nodeları paralel kopyalama başlatılıyor..."
  rsync -a --info=progress2 --ignore-existing /comfyui/models/ "${VOL_ROOT}/models/" &
  RSYNC_PID_1=$!
  
  rsync -a --info=progress2 --ignore-existing /comfyui/custom_nodes/ "${VOL_ROOT}/custom_nodes/" &
  RSYNC_PID_2=$!
  
  # Kopyalama işlemlerini bekle
  echo "⏳ Beklenilen işlemler: $RSYNC_PID_1, $RSYNC_PID_2"
  wait $RSYNC_PID_1
  echo "✓ Model kopyalama tamamlandı"
  wait $RSYNC_PID_2
  echo "✓ Custom node kopyalama tamamlandı"
  
  # Sistem optimizasyonları - yüksek bellek sistemleri için
  echo "vm.max_map_count=2097152" > "${VOL_ROOT}/sysctl.conf"
  echo "vm.overcommit_memory=1" >> "${VOL_ROOT}/sysctl.conf"
  echo "vm.swappiness=1" >> "${VOL_ROOT}/sysctl.conf"
  
  # Network optimizasyonları
  echo "net.core.rmem_max=16777216" >> "${VOL_ROOT}/sysctl.conf"
  echo "net.core.wmem_max=16777216" >> "${VOL_ROOT}/sysctl.conf"
  
  touch "${VOL_ROOT}/.ready"
  echo "🟢  Volume hazırlaması tamamlandı."
  
  # IO önbelleğini yeniden temizle
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
fi

###############################################################################
# 4)   COMFYUI + RUNPOD HANDLER İÇİN YÜKSEK PERFORMANS AYARLARI
###############################################################################
# Ultra yüksek performans için bellek ayarlarını uygula
echo "💪 Yüksek performans bellek ayarları uygulanıyor..."

# Bellek sınırlamaları uzaklaştırılıyor
ulimit -m unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
ulimit -l unlimited 2>/dev/null || true

# Disk IO performansı
echo "⚡ IO performansı artırılıyor..."
# readahead değerini artır (daha hızlı disk okuma)
blockdev --setra 16384 /dev/sda 2>/dev/null || true

# HuggingFace önbellek boyutunu artır
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface
mkdir -p /tmp/huggingface

###############################################################################
# 5)   COMFYUI + RUNPOD HANDLER (YÜKSEK PERFORMANS MODU)
###############################################################################
echo "🚀  Yüksek Performans Modunda ComfyUI başlatılıyor..."

# ComfyUI için yüksek bellek parametreleri
python3 /comfyui/main.py \
        --listen "${COMFY_HOST}" \
        --port "${COMFY_PORT}" \
        --disable-auto-launch \
        --disable-metadata \
        --highvram \
        --preview-method auto \
        --gpu-only \
        --force-fp16 &

# Yüksek güvenilirlik için ComfyUI'nin başlamasını bekle
COMFY_URL="http://${COMFY_HOST}:${COMFY_PORT}"
MAX_RETRY=20
echo "⏳ ComfyUI hazır olana kadar bekleniyor (timeout: ${MAX_RETRY}s)..."

for i in $(seq 1 $MAX_RETRY); do
  if curl -s "$COMFY_URL" >/dev/null 2>&1; then
    echo "✅  ComfyUI hazır: ${COMFY_URL}"
    break
  fi
  
  if [ "$i" -lt "$MAX_RETRY" ]; then
    sleep 1
  else
    echo "⚠️  ComfyUI'nin başlaması bekleniyor, devam ediliyor..."
  fi
done

# Yüksek bellek sistemleri için RunPod handler ayarları
export RP_CUDA_MEMORY_STRATEGY=expansive
export RP_POLLING_MAX_RETRIES=1000
export RP_POLLING_INTERVAL_MS=100

# RunPod handler'ı başlat
if [[ "${SERVE_API_LOCALLY}" == "true" ]]; then
  echo "🔌  Yüksek Performans: RunPod handler (yerel API) başlatılıyor..."
  exec python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  echo "🔌  Yüksek Performans: RunPod handler başlatılıyor..."
  exec python3 -u /rp_handler.py
fi