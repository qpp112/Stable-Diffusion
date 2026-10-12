#!/bin/bash
# Reproduce the README numbers on HPC3:   sbatch hpc3/benchmark.slurm
# free-gpu costs nothing but jobs can be killed (preempted) at any time; just resubmit.
# Each script below writes its own results, so a killed job only loses the step that was running.
# Max 9 GB host RAM per CPU core on free-gpu, so --mem must stay <= 9G x cpus-per-task.
#SBATCH --job-name=sd-scratch-bench
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1                 # pin a type for comparable numbers, e.g. gpu:A30:1 or gpu:V100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G                    # fp32 SDXL baseline needs ~14 GB host RAM to load
#SBATCH --time=08:00:00              # free-gpu allows up to 3 days; first run also downloads ~11 GB
#SBATCH --output=slurm-%x-%j.out

module load python/3.10.2            # before `set -u`: Lmod scripts can trip on unset variables
set -uo pipefail                     # no -e: one failed step should not stop the others
source ${ENV_DIR:-/pub/$USER/envs/sdscratch}/bin/activate
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m pytest -q tests
python scripts/toy_gmm.py
python scripts/benchmark_memory.py --model sdxl --size 1024 --steps 20
python scripts/benchmark_memory.py --model sdxl --size 1024 --steps 20 --vram-limit-gb 8 --skip-baseline
python scripts/benchmark_memory.py --model sd15 --size 1024 --steps 20
python scripts/compare_samplers.py --model sd15 --nfe 5 10 20 50
python scripts/compare_samplers.py --model sdxl --nfe 10 20 30 --reference-nfe 200
