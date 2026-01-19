#!/bin/bash
# Quick setup script for A100 server

set -e

echo "Installing dependencies..."

# System deps (run with sudo if needed)
apt update && apt install -y \
  build-essential curl git libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
  liblzma-dev tk-dev ca-certificates

# Pyenv - check if already installed
if [ -d "$HOME/.pyenv" ]; then
    echo "pyenv already installed, skipping..."
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
else
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

source ~/.bashrc
# Python 3.12 - check if already installed
if pyenv versions | grep -q "3.12"; then
    echo "Python 3.12 already installed, skipping..."
else
    echo "Installing Python 3.12..."
    pyenv install 3.12.12
fi
pyenv global 3.12.12

source ~/.bashrc

# Pip packages
pip install -U pip setuptools wheel

# PyTorch
pip install psutil
pip install torch torchvision torchaudio

# Flash Attention
pip install flash-attn --no-build-isolation

# Training deps
pip install sentence-transformers datasets transformers wandb huggingface-hub tqdm dotenv distil-trainer

echo ""
echo "Setup complete!"
echo ""
echo "Now set your tokens:"
echo "  export HF_TOKEN='your_huggingface_token'"
echo "  export WANDB_API_KEY='your_wandb_key'"
echo ""
echo "Then run:"
echo "  bash run_training.sh or nohup python train_magibu_cosmos.py > train.log 2>&1 &"
