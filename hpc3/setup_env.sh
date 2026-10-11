#!/bin/bash
# One-time environment setup on UCI HPC3 (run on a login node).
# Module names change over time - check `module avail python` / `module avail anaconda` first.
set -euo pipefail

module load python/3.10.2 2>/dev/null || module load anaconda 2>/dev/null || true

ENV_DIR=${ENV_DIR:-$HOME/envs/sdscratch}
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# PyTorch wheels bundle the CUDA runtime; only the node's NVIDIA driver is needed.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e "$(dirname "$0")/.." pytest regex matplotlib

# Model weights are large (SDXL ~7 GB): keep the Hugging Face cache off your home quota.
echo "export HF_HOME=/pub/$USER/hf_cache" >> "$ENV_DIR/bin/activate"
echo "Done. Activate with: source $ENV_DIR/bin/activate"
