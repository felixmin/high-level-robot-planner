# LAPA Project Dockerfile
# Base image: NVIDIA PyTorch with CUDA 12.1

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Metadata
LABEL maintainer="LAPA Team"
LABEL description="Container for LAPA robot learning system"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /workspace/
COPY packages/ /workspace/packages/
COPY config/ /workspace/config/
COPY scripts/ /workspace/scripts/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.1.0 \
    torchvision>=0.16.0 \
    pytorch-lightning>=2.1.0 \
    lightning-fabric>=2.1.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    webdataset>=0.2.86 \
    transformers>=4.35.0 \
    timm>=0.9.0 \
    accelerate>=0.25.0 \
    wandb>=0.16.0 \
    tensorboard>=2.15.0 \
    tqdm>=4.66.0 \
    einops>=0.7.0 \
    rich>=13.7.0

# Install project packages in editable mode
RUN pip install -e /workspace/packages/common && \
    pip install -e /workspace/packages/laq && \
    pip install -e /workspace/packages/foundation && \
    pip install -e /workspace/packages/low_level

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["/bin/bash"]

