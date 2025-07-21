# Use NVIDIA CUDA 11.8 base image with Ubuntu 20.04
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=11.8
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libc6-dev \
    vim \
    nano \
    htop \
    unzip \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash

# Create conda environment for MonSter with Python 3.8
RUN /opt/conda/bin/conda create -n monster python=3.8 -y && \
    echo "source activate monster" > ~/.bashrc

# Copy requirements file
COPY .devcontainer/requirements.txt /tmp/requirements.txt

# Activate the conda environment and install dependencies
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/bin/activate monster && \
    # Install PyTorch with CUDA 11.8 support first
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    # Install mmcv with specific CUDA build
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && \
    # Install remaining dependencies from requirements file
    pip install -r /tmp/requirements.txt

# Set up the working directory
WORKDIR /workspace

# Copy the project files (this will be mounted in devcontainer)
COPY . /workspace/

# Set the default conda environment
RUN echo "conda activate monster" >> ~/.bashrc

# Create directories that might be needed
RUN mkdir -p /workspace/checkpoints && \
    mkdir -p /workspace/pretrained && \
    mkdir -p /workspace/datasets

# Set permissions
RUN chmod -R 755 /workspace

# Expose ports for Jupyter, TensorBoard, and Gradio
EXPOSE 8888 6006 7860

# Set the default command
CMD ["/bin/bash"]