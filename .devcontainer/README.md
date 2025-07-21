# MonSter Development Container

This directory contains the development container configuration for the MonSter project, providing a ready-to-use GPU-enabled environment with CUDA 11.8 support.

## 🚀 Quick Start

### Prerequisites
- Docker with GPU support (Docker Desktop with WSL2 or Linux with nvidia-docker2)
- NVIDIA GPU with compatible drivers
- Visual Studio Code with the "Dev Containers" extension

### Using the Development Container

1. **Open in VS Code**: Open the project folder in VS Code
2. **Reopen in Container**: When prompted, click "Reopen in Container" or run the command palette (`Ctrl+Shift+P`) and select "Dev Containers: Reopen in Container"
3. **Wait for Build**: The container will build automatically (this may take 15-20 minutes the first time)
4. **Verify Setup**: Run the setup check script:
   ```bash
   ./setup_check.sh
   ```

## 🔧 Configuration Details

### Base Image
- **NVIDIA CUDA 11.8** with Ubuntu 20.04
- **Python 3.8** in a conda environment named `monster`

### GPU Support
- Configured with `--gpus=all` for full GPU access
- Increased shared memory (`--shm-size=16gb`) for multi-GPU training
- Environment variables set for NVIDIA GPU visibility

### Dependencies Included
- **PyTorch 2.0.1** with CUDA 11.8 support
- **Core libraries**: timm==0.6.13, mmcv==2.1.0, accelerate==1.0.1
- **Vision libraries**: opencv-python, scikit-image, matplotlib
- **Development tools**: Jupyter, tensorboard, gradio
- **Code quality**: flake8, black, isort, pytest

### VS Code Extensions
- Python development tools
- Jupyter notebook support
- NVIDIA Nsight for GPU debugging
- Code formatting and linting

### Port Forwarding
- **8888**: Jupyter Lab
- **6006**: TensorBoard
- **7860**: Gradio demos

## 📁 Directory Structure

```
.devcontainer/
├── devcontainer.json    # Main configuration file
├── Dockerfile          # Container build instructions
├── requirements.txt    # Python dependencies
├── setup_check.sh      # Environment verification script
└── README.md          # This file
```

## 🏃‍♂️ Running MonSter

### Training
```bash
# Activate the conda environment (if not already activated)
conda activate monster

# Single GPU
python train_kitti.py

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_kitti.py
```

### Evaluation
```bash
python evaluate_stereo.py --restore_ckpt ./pretrained/sceneflow.pth --dataset kitti
```

### Development Tools
```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Start TensorBoard
tensorboard --logdir=./checkpoints --host=0.0.0.0 --port=6006

# Run the demo
python demo_video.py
```

## ⚠️ Troubleshooting

### GPU Not Detected
1. Ensure NVIDIA drivers are installed on the host
2. Install nvidia-docker2 on Linux or enable GPU support in Docker Desktop
3. Check that `nvidia-smi` works on the host system

### Out of Memory Errors
1. Reduce batch size in training configurations
2. Use gradient checkpointing
3. Ensure sufficient GPU memory is available

### Build Failures
1. Check internet connection for downloading packages
2. Ensure sufficient disk space (container needs ~10GB)
3. Try rebuilding with "Dev Containers: Rebuild Container"

## 🔗 Useful Commands

```bash
# Check CUDA and GPU status
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run environment check
./setup_check.sh
```

## 📝 Customization

To modify the environment:

1. **Add Dependencies**: Edit `requirements.txt` and rebuild the container
2. **Change Python Version**: Modify the `conda create` command in `Dockerfile`
3. **Add System Packages**: Add them to the `apt-get install` command in `Dockerfile`
4. **Configure VS Code**: Edit the `customizations.vscode` section in `devcontainer.json`

Remember to rebuild the container after making changes to the configuration files.