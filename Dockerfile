FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Install core libraries
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/Zurich" \
    apt-get install -y software-properties-common wget git vim ffmpeg tmux \
    python3-pip unzip curl python3-venv libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential python3-dev cmake ninja-build protobuf-compiler libprotobuf-dev

# Create the cache folders
RUN mkdir -p /app/tmp /app/tmp/torch_cache

# Define the PATH environment variable
ENV TORCH_HOME=/app/tmp/torch_cache \
    TMPDIR=/app/tmp \
    TORCHINDUCTOR_CACHE_DIR=/app/tmp/torch_cache 

# Set environment variables globally
RUN echo "export TMPDIR=/app/tmp" >> /etc/environment && \
    echo "export TORCH_HOME=/app/tmp/torch_cache" >> /etc/environment && \
    echo "export NO_ALBUMENTATIONS_UPDATE=1" >> /etc/environment && \
    echo "export TORCHINDUCTOR_CACHE_DIR=/app/tmp/torch_cache" >> /etc/environment && \
    echo "root:x:0:0:root:/root:/bin/bash" >> /etc/passwd && \
    chmod -R 777 $TMPDIR

# Install uv (for pip downloads in parallel)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv pip install --break-system-packages --system --upgrade pip setuptools
ENV UV_HTTP_TIMEOUT=300

# Install TensorRT
RUN uv pip install --break-system-packages --system --upgrade tensorrt

# Copy the rest of the repository
COPY ./ /app/

# Activate environment and install repository libraries
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV NO_ALBUMENTATIONS_UPDATE="1"
RUN uv pip install --break-system-packages --system -r /app/requirements-pytorch.txt
RUN uv pip install --break-system-packages --system -r /app/requirements.txt
# Fix bug
RUN uv pip install --break-system-packages --system albumentationsx==2.0.13

# Change permissions of the entire /app/ directory to be writable by all users
WORKDIR /app/
RUN chmod -R 777 /app/
