#!/bin/bash

# MonSter Development Environment Setup Script
echo "🚀 Setting up MonSter development environment..."

# Check CUDA availability
echo "📋 Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available')
"

# Check key dependencies
echo "📦 Checking key dependencies..."
python -c "
try:
    import torchvision
    print(f'✅ torchvision: {torchvision.__version__}')
except ImportError:
    print('❌ torchvision not found')

try:
    import cv2
    print(f'✅ opencv-python: {cv2.__version__}')
except ImportError:
    print('❌ opencv-python not found')

try:
    import matplotlib
    print(f'✅ matplotlib: {matplotlib.__version__}')
except ImportError:
    print('❌ matplotlib not found')

try:
    import timm
    print(f'✅ timm: {timm.__version__}')
except ImportError:
    print('❌ timm not found')

try:
    import mmcv
    print(f'✅ mmcv: {mmcv.__version__}')
except ImportError:
    print('❌ mmcv not found')

try:
    import accelerate
    print(f'✅ accelerate: {accelerate.__version__}')
except ImportError:
    print('❌ accelerate not found')

try:
    import gradio
    print(f'✅ gradio: {gradio.__version__}')
except ImportError:
    print('❌ gradio not found')
"

# Test GPU memory allocation
echo "🧪 Testing GPU functionality..."
python -c "
import torch
if torch.cuda.is_available():
    try:
        x = torch.randn(2, 3, 256, 256).cuda()
        y = torch.nn.functional.relu(x)
        print('✅ GPU tensor operations working')
        print(f'   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
        print(f'   GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
    except Exception as e:
        print(f'❌ GPU operations failed: {e}')
else:
    print('⚠️  No GPU available for testing')
"

echo "🎉 Setup verification complete!"
echo ""
echo "📚 Quick start:"
echo "   • Run training: CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_kitti.py"
echo "   • Run evaluation: python evaluate_stereo.py --restore_ckpt ./pretrained/sceneflow.pth --dataset kitti"
echo "   • Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "   • Start TensorBoard: tensorboard --logdir=./checkpoints --host=0.0.0.0 --port=6006"