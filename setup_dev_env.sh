#!/bin/bash
set -e  # Exit on first error
set -o pipefail

# ------------------------------
# 0. Check Miniconda
# ------------------------------
if [ ! -d "$HOME/miniconda3" ]; then
    echo "⏳ Installing Miniconda..."
    mkdir -p "$HOME/miniconda3"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
    bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
else
    echo "✅ Miniconda already installed."
fi

# ------------------------------
# 1. Initialize conda/mamba
# ------------------------------
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda install -y mamba -n base -c conda-forge

# ------------------------------
# 2. Create and activate environment
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
# 3. Install uv and dependencies
# ------------------------------
echo "⏳ Installing project dependencies with uv..."
uv pip install --editable ".[dev,training,evaluation]"

# ------------------------------
# 4. Optional: Flash Attention (CUDA)
# ------------------------------
read -p "Install flash-attn for GPU support? [y/N]: " install_flash
if [[ "$install_flash" == "y" || "$install_flash" == "Y" ]]; then
    echo "⏳ Installing flash-attn..."
    pip install ninja packaging
    MAX_JOBS=4 pip install flash-attn --no-build-isolation
fi

# ------------------------------
# 5. Optional: VLLM nightly (GPU)
# ------------------------------
read -p "Install vllm nightly GPU package? [y/N]: " install_vllm
if [[ "$install_vllm" == "y" || "$install_vllm" == "Y" ]]; then
    echo "⏳ Installing vllm nightly..."
    pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm==0.10.2
fi

# ------------------------------
# 6. Final messages
# ------------------------------
echo "✅ Setup complete. Environment '$ENV_NAME' is ready."
echo "ℹ️ Make sure to activate: conda activate $ENV_NAME"
echo "ℹ️ You can now run:"
echo "    make setup        # create folders + .env"
echo "    make prepare-data # prepare datasets"
echo "    make train        # train model"
echo "    make evaluate     # evaluate"
