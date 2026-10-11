#!/bin/bash
# Reproduce the README numbers on HPC3:   sbatch hpc3/benchmark.slurm
# free-gpu jobs are free but can be preempted; use -p gpu with your lab's account for guaranteed runs.
#SBATCH --job-name=sd-scratch-bench
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1                 # e.g. gpu:V100:1 or gpu:A30:1 to pin a GPU type
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                    # fp32 SDXL baseline needs ~14 GB host RAM to load
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source ${ENV_DIR:-$HOME/envs/sdscratch}/bin/activate
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m pytest -q tests
python scripts/toy_gmm.py
python scripts/benchmark_memory.py --model sdxl --size 1024 --steps 20
python scripts/benchmark_memory.py --model sdxl --size 1024 --steps 20 --vram-limit-gb 8 --skip-baseline
python scripts/benchmark_memory.py --model sd15 --size 1024 --steps 20
python scripts/compare_samplers.py --model sd15 --nfe 5 10 20 50
python scripts/compare_samplers.py --model sdxl --nfe 10 20 30 --reference-nfe 200
