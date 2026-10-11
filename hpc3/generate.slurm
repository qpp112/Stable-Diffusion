#!/bin/bash
# Usage: sbatch hpc3/generate.slurm "a watercolor fox in a snowy forest"
#SBATCH --job-name=sd-scratch-gen
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source ${ENV_DIR:-$HOME/envs/sdscratch}/bin/activate
cd "$SLURM_SUBMIT_DIR"
python scripts/generate.py --model sdxl --prompt "${1:-a lighthouse on a cliff at dusk, oil painting}" \
    --steps 25 --num 4 --preset balanced --out outputs/hpc3
