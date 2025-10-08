#!/bin/bash
set -e  # Exit on first error
set -o pipefail

# ------------------------------
# 0. Clone repository and enter it
# ------------------------------
REPO_URL="https://github.com/chrisjcc/text-to-sql-finetuning.git"
REPO_DIR="$(basename "$REPO_URL" .git)"

if [ ! -d "$REPO_DIR" ]; then
    echo "⏳ Cloning repository from $REPO_URL..."
    git clone "$REPO_URL"
else
    echo "✅ Repository '$REPO_DIR' already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
fi

cd "$REPO_DIR"
echo "📂 Entered project directory: $(pwd)"

# ------------------------------
# 1. Check Miniconda
# ------------------------------
ARCH=$(uname -m)

# Normalize architecture name to match Miniconda's naming convention
case "$ARCH" in
    x86_64)   ARCH_TAG="x86_64" ;;
    aarch64)  ARCH_TAG="aarch64" ;;
    arm64)    ARCH_TAG="aarch64" ;;  # in case uname returns arm64 (common on Macs)
    *) echo "❌ Unsupported architecture: $ARCH" && exit 1 ;;
esac

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH_TAG}.sh"

if [ ! -d "$HOME/miniconda3" ]; then
    echo "⏳ Installing Miniconda for ${ARCH_TAG}..."
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

conda install -y mamba -n base -c conda-forge

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
echo "✅ Setup complete. Environment '$ENV_NAME' is ready."
echo "ℹ️ Make sure to activate: conda activate $ENV_NAME"
echo "ℹ️ You can now run:"
echo "    make setup        # create folders + .env"
echo "    make prepare-data # prepare datasets"
echo "    make train        # train model"
echo "    make evaluate     # evaluate"
