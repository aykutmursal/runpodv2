###############################################################################
# Stage 0 ▸ base : CUDA 12.4 + ComfyCLI + helper script’ler
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
      google-perftools ca-certificates && \
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
# Stage 1 ▸ models + custom-nodes  (TEK RUN → TEK LAYER)
###############################################################################
FROM base AS build_models

ARG HF_TOKEN=""
ARG CIVI_TOKEN=""

RUN set -ex \
 #── Diffusion modelleri
 && mkdir -p /comfyui/models/diffusion_models \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/diffusion_models/hidream_i1_dev_bf16.safetensors \
      "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_dev_bf16.safetensors?download=true&token=${HF_TOKEN}" \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      --header="Authorization: Bearer ${CIVI_TOKEN}" \
      -O /comfyui/models/diffusion_models/fluxFillFP8_v10.safetensors \
      "https://civitai.com/api/download/models/1085456?type=Model&format=SafeTensor&size=full&fp=fp8&download=true" \
 \
 #── Checkpoint
 && mkdir -p /comfyui/models/checkpoints \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/checkpoints/flux1-dev-fp8.safetensors \
      "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors?download=true&token=${HF_TOKEN}" \
 \
 #── LoRA
 && mkdir -p /comfyui/models/loras \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/loras/comfyui_portrait_lora64.safetensors \
      "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors?download=true&token=${HF_TOKEN}" \
 \
 #── VAE
 && mkdir -p /comfyui/models/vae/FLUX1 \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/vae/FLUX1/ae.safetensors \
      "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors?download=true&token=${HF_TOKEN}" \
 \
 #── Text encoders
 && mkdir -p /comfyui/models/text_encoders \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/text_encoders/clip_l.safetensors \
      "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_l.safetensors?download=true&token=${HF_TOKEN}" \
 && wget -q --continue --retry-connrefused --waitretry=5 -t 5 \
      -O /comfyui/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors \
      "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors?download=true&token=${HF_TOKEN}" \
 \
 #── Custom nodes (9 depo)
 && mkdir -p /comfyui/custom_nodes && cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack comfyui-impact-pack \
 && pip install --no-cache-dir -r comfyui-impact-pack/requirements.txt \
 && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
 && pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2.git \
 && pip install --no-cache-dir -r ComfyUI-Florence2/requirements.txt \
 && git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git \
 && git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache.git \
 && pip install --no-cache-dir -r ComfyUI-TeaCache/requirements.txt \
 && git clone --depth 1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git \
 && git clone --depth 1 https://github.com/aria1th/ComfyUI-LogicUtils.git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# Stage 2 ▸ final : entrypoint + API-only
###############################################################################
FROM build_models AS final

ENV MODEL_TYPE=dev-bf16
ENTRYPOINT ["/start.sh"]   # rsync + 127.0.0.1:8188 (UI dışa kapalı)
CMD []
