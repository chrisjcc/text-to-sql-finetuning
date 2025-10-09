#!/bin/bash
set -e  # Exit on first error
set -o pipefail

# ------------------------------
# 0. Preliminary checks
# ------------------------------
ARCH=$(uname -m)

if [ "$ARCH" != "x86_64" ]; then
    echo "❌ Unsupported architecture: $ARCH"
    echo "This repository requires x86_64 (NVIDIA GPU support expected)."
    exit 1
fi

echo "✅ Architecture check passed: $ARCH"

# Confirm we are inside the repository
if [ ! -f "setup_dev_env.sh" ]; then
    echo "❌ This script must be run from within the repository root."
    exit 1
fi

echo "📂 Running setup inside repository: $(pwd)"

# ------------------------------
# 1. Check and install Miniconda
# ------------------------------
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

if [ ! -d "$HOME/miniconda3" ]; then
    echo "⏳ Installing Miniconda for x86_64..."
    mkdir -p "$HOME/miniconda3"
    wget -q "$MINICONDA_URL" -O "$HOME/miniconda3/miniconda.sh"
    bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
else
    echo "✅ Miniconda already installed."
fi

# ------------------------------
# 2. Initialize conda/mamba
# ------------------------------
# This ensures the conda command is initialized and available in the current shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accept Anaconda Terms of Service (required for repo.anaconda.com channels)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Check if mamba is installed in the base environment
if conda list -n base | grep -q '^mamba\s'; then
    echo "✅ Mamba is already installed in the base environment."
else
    echo "⏳ Installing mamba in the base environment..."
    conda install -y mamba -n base -c conda-forge
fi

# ------------------------------
# 3. Create and activate environment
# ------------------------------
ENV_NAME="llm_env"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ Conda environment '$ENV_NAME' already exists."
else
    echo "⏳ Creating conda environment '$ENV_NAME'..."
    mamba env create -f environment.yml
fi

echo "ℹ️ Activate the environment with: conda activate $ENV_NAME"
conda activate "$ENV_NAME"

# ------------------------------
# 4. Install uv and dependencies
# ------------------------------
echo "⏳ Installing project dependencies with uv..."
uv pip install --editable ".[dev,training,evaluation]"

# ------------------------------
# 5. Optional: Flash Attention (CUDA)
# ------------------------------
read -p "Install flash-attn for GPU support? [y/N]: " install_flash
if [[ "$install_flash" == "y" || "$install_flash" == "Y" ]]; then
    echo "⏳ Installing flash-attn..."
    pip install ninja packaging
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
fi

# ------------------------------
# 6. Optional: VLLM nightly (GPU)
# ------------------------------
read -p "Install vllm nightly GPU package? [y/N]: " install_vllm
if [[ "$install_vllm" == "y" || "$install_vllm" == "Y" ]]; then
    echo "⏳ Installing vllm nightly..."
    pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm==0.10.2
fi

# ------------------------------
# 7. Final messages
# ------------------------------
echo ""
echo "✅ Setup complete. Environment '$ENV_NAME' is ready."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Restart your shell to apply conda initialization:"
echo "    exec bash"
echo ""
echo "2️⃣  OR source your shell config file:"
echo "    source ~/.bashrc    # for bash"
echo "    source ~/.zshrc     # for zsh"
echo ""
echo "3️⃣  Activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "    OR use the helper script:"
echo "    source activate_env.sh"
echo ""
echo "4️⃣  Run your commands:"
echo "    make setup        # create folders + .env"
echo "    make prepare-data # prepare datasets"
echo "    make train        # train model"
echo "    make evaluate     # evaluate"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
