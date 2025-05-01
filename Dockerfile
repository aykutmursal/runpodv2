###############################################################################
# Stage 0 ▸ base : CUDA 12.4 + ComfyCLI + helper script'ler
###############################################################################
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# — system deps —
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-distutils python3-dev \
      build-essential git wget rsync libgl1 libglib2.0-0 libsm6 libxrender1 \
      google-perftools ca-certificates curl aria2 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3  /usr/bin/pip  && \
    rm -rf /var/lib/apt/lists/*

# — ComfyCLI & runtime —
RUN python3 -m pip install --no-cache-dir comfy-cli==1.3.8 runpod requests && \
    yes | comfy --workspace /comfyui install --cuda-version 12.4 --nvidia

# — helper dosyalar —
ADD src/extra_model_paths.yaml /
WORKDIR /
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json /
RUN chmod +x /start.sh /restore_snapshot.sh

###############################################################################
# Stage 1 ▸ models + custom-nodes (ayrı katmanlarda)
###############################################################################
FROM base AS build_models

ARG HF_TOKEN
ARG CIVI_TOKEN

# --- Civitai modeli (özellikle önemli olanı önce indirelim) ---
RUN mkdir -p /comfyui/models/diffusion_models && \
    echo "Downloading fluxFillFP8_v10.safetensors from Civitai..." && \
    aria2c --max-tries=10 --retry-wait=10 --timeout=60 --connect-timeout=60 --allow-overwrite=true \
      --split=8 --max-connection-per-server=8 --dir=/comfyui/models/diffusion_models \
      --out=fluxFillFP8_v10.safetensors \
      --header="Authorization: Bearer ${CIVI_TOKEN}" \
      "https://civitai.com/api/download/models/1085456?type=Model&format=SafeTensor&size=full&fp=fp8" && \
    [ -f "/comfyui/models/diffusion_models/fluxFillFP8_v10.safetensors" ] && \
    echo "Successfully downloaded fluxFillFP8_v10.safetensors"

# --- Diffusion model 1: hidream_i1_dev_bf16 ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    mkdir -p /comfyui/models/diffusion_models && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/diffusion_models/hidream_i1_dev_bf16.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_dev_bf16.safetensors?download=true&token=${HF_TOKEN}" && \
    [ -f "/comfyui/models/diffusion_models/hidream_i1_dev_bf16.safetensors" ] && \
    echo "Successfully downloaded hidream_i1_dev_bf16.safetensors"

# --- Text encoders - İstediğiniz formatla ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    mkdir -p /comfyui/models/text_encoders && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/clip_l_hidream.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_l_hidream.safetensors?download=true&token=${HF_TOKEN}" && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/clip_g_hidream.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_g_hidream.safetensors?download=true&token=${HF_TOKEN}" && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors?download=true&token=${HF_TOKEN}" && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors?download=true&token=${HF_TOKEN}"

# --- Orijinal text encoder dosyalarını da ekleyelim ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/clip_l.safetensors \
      "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_l.safetensors?download=true&token=${HF_TOKEN}" && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors \
      "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors?download=true&token=${HF_TOKEN}"

# --- VAE - İstediğiniz formatla ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    mkdir -p /comfyui/models/vae && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/vae/ae.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/vae/ae.safetensors?download=true&token=${HF_TOKEN}"

# --- VAE - FLUX1 ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    mkdir -p /comfyui/models/vae/FLUX1 && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/vae/FLUX1/ae.safetensors \
      "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors?download=true&token=${HF_TOKEN}"

# --- LoRA ---
RUN --mount=type=cache,target=/tmp/wget-cache \
    mkdir -p /comfyui/models/loras && \
    wget --continue --retry-connrefused --waitretry=10 -t 10 --timeout=60 \
      -O /comfyui/models/loras/comfyui_portrait_lora64.safetensors \
      "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors?download=true&token=${HF_TOKEN}"

###############################################################################
# Custom nodes (her biri ayrı katmanda)
###############################################################################
# --- Custom nodes (impact-pack) ---
RUN mkdir -p /comfyui/custom_nodes && cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack comfyui-impact-pack && \
    pip install --no-cache-dir -r comfyui-impact-pack/requirements.txt

# --- Custom nodes (rgthree) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git

# --- Custom nodes (KJNodes) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt

# --- Custom nodes (Florence2) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2.git && \
    pip install --no-cache-dir -r ComfyUI-Florence2/requirements.txt

# --- Custom nodes (essentials) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git

# --- Custom nodes (TeaCache) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache.git && \
    pip install --no-cache-dir -r ComfyUI-TeaCache/requirements.txt

# --- Custom nodes (Inpaint-CropAndStitch and LogicUtils) ---
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git && \
    git clone --depth 1 https://github.com/aria1th/ComfyUI-LogicUtils.git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Dosyaların varlığını kontrol et ---
RUN ls -la /comfyui/models/diffusion_models/ && \
    ls -la /comfyui/models/text_encoders/ && \
    ls -la /comfyui/models/vae/ && \
    ls -la /comfyui/models/vae/FLUX1/ && \
    ls -la /comfyui/models/loras/

###############################################################################
# Stage 2 ▸ final : entrypoint + API-only
###############################################################################
FROM build_models AS final

ENV MODEL_TYPE=dev-bf16
ENTRYPOINT ["/start.sh"]   # rsync + 127.0.0.1:8188 (UI dışa kapalı)
CMD []