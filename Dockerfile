# Host NVIDIA driver is NOT part of this image. The container requires a
# compatible host driver and the NVIDIA Container Toolkit.
# The base tag is pinned; override with --build-arg CUDA_IMAGE=... if needed.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/compute-topk-scores

COPY requirements/core.txt requirements/core.txt
RUN python3 -m pip install --upgrade pip==26.2.2 \
    && python3 -m pip install -r requirements/core.txt \
    && python3 -c "import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)"

COPY . .

RUN python3 scripts/check_env.py --strict

CMD ["python3", "scripts/check_env.py"]
