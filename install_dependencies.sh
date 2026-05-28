#!/bin/bash

# 安装系统依赖
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y hmmer python3-pip

# 安装Python依赖
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# 验证安装
echo "Verifying installations..."
which hmmscan
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# 下载预训练模型（如果需要）
echo "Checking for pre-trained models..."
if [ ! -d "model/esm2_t33_650M_UR50D" ]; then
    echo "Downloading ESM-2 pre-trained model..."
    mkdir -p model
    cd model
    git clone https://huggingface.co/facebook/esm2_t33_650M_UR50D
    cd ..
fi

echo "All dependencies installed successfully!"
