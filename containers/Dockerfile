FROM nvcr.io/nvidia/pytorch:25.10-py3

# Configure environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl

# Install system dependencies (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    libglib2.0-0 \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    ffmpeg \
    pkg-config \
    && useradd --create-home --shell /bin/bash user_lerobot \
    && usermod -aG sudo user_lerobot \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/high-level-robot-planner
RUN chown -R user_lerobot:user_lerobot /workspace

# Switch to the non-root user
USER user_lerobot

# Environment variables
ARG CONDA_ENV_NAME=hlrp
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/bin:$PATH
ENV HOME=/home/user_lerobot \
    HF_HOME=/home/user_lerobot/.cache/huggingface \
    HF_LEROBOT_HOME=/home/user_lerobot/.cache/huggingface/lerobot \
    TORCH_HOME=/home/user_lerobot/.cache/torch \
    TRITON_CACHE_DIR=/home/user_lerobot/.cache/triton

# Preserve the PyTorch stack from the NGC base image:
# 1) clone base env (keeps torch/torchvision/torchaudio versions)
# 2) install cmake<4 + ninja
# 3) install egl_probe
# 4) install lerobot + libero with torch packages pinned via constraints
# Policy plugin is installed at runtime from the mounted repo.
RUN conda create -y -n ${CONDA_ENV_NAME} --clone base && \
    conda install -y -n ${CONDA_ENV_NAME} -c conda-forge "cmake<4" ninja && \
    conda run -n ${CONDA_ENV_NAME} python -c "import importlib.util, importlib; \
pkgs=['torch','torchvision','torchaudio']; \
lines=[]; \
[lines.append(f'{p}=={importlib.import_module(p).__version__}') for p in pkgs if importlib.util.find_spec(p) is not None]; \
open('/tmp/torch-constraints.txt','w',encoding='utf-8').write('\\n'.join(lines) + ('\\n' if lines else ''))" && \
    conda run -n ${CONDA_ENV_NAME} pip install --no-build-isolation egl_probe && \
    conda run -n ${CONDA_ENV_NAME} pip install --constraint /tmp/torch-constraints.txt "lerobot[libero]" && \
    conda clean -afy && \
    echo "conda activate ${CONDA_ENV_NAME}" >> /home/user_lerobot/.bashrc


# Set the default command
CMD ["/bin/bash"]
