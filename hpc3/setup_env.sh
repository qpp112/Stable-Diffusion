#!/bin/bash
# One-time environment setup on UCI HPC3. Same steps as hpc3/README.md, in one script.
# Everything large (venv, pip cache, model weights) goes to /pub/$USER: $HOME has a 50 GB quota.
# Run it on an interactive node, not the login node:   srun -c 4 -p free --mem=16G --pty /bin/bash -i
# Module names change over time - check `module avail python` first.
set -euo pipefail

module load python/3.10.2

ENV_DIR=${ENV_DIR:-/pub/$USER/envs/sdscratch}
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
export PIP_CACHE_DIR=/pub/$USER/.cache/pip   # pip's cache of the CUDA wheels is several GB
pip install --upgrade pip
# PyTorch wheels bundle the CUDA runtime; only the node's NVIDIA driver is needed.
# cu126 wheels still support the V100s in free-gpu (newer CUDA 13 builds drop Volta).
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e "$(dirname "$0")/..[dev]"

# Model weights are large (SDXL ~7 GB): keep the Hugging Face cache off your home quota.
mkdir -p "/pub/$USER/hf_cache"
echo "export HF_HOME=/pub/$USER/hf_cache" >> "$ENV_DIR/bin/activate"
echo "Done. Activate with: source $ENV_DIR/bin/activate"
