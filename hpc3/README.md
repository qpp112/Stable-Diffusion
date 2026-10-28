# Running on UCI HPC3

Login needs the campus network or the UCI VPN, plus a DUO push (or an SSH key with a passphrase).
`free-gpu` is free for everyone but jobs can be preempted; `gpu` needs a lab GPU account.
Docs: https://rcic.uci.edu/slurm/slurm.html

## One-time setup

`$HOME` has a 50 GB quota and PyTorch's CUDA wheels plus pip's cache take several GB, so the
virtualenv, the pip cache, the model weights (~11 GB) and the repo itself all live in `/pub/$USER`.

```bash
ssh <ucinetid>@hpc3.rcic.uci.edu
srun -c 4 -p free --mem=16G --pty /bin/bash -i        # build on a compute node, not the login node

echo 'export PIP_CACHE_DIR=/pub/$USER/.cache/pip' >> ~/.bashrc
echo 'export HF_HOME=/pub/$USER/hf_cache' >> ~/.bashrc
echo 'export ENV_DIR=/pub/$USER/envs/sdscratch' >> ~/.bashrc
source ~/.bashrc
mkdir -p /pub/$USER/envs $PIP_CACHE_DIR $HF_HOME

module load python/3.10.2
cd /pub/$USER
git clone https://github.com/qpp112/Stable-Diffusion.git
cd Stable-Diffusion
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
exit
```

(`bash hpc3/setup_env.sh` does the same thing.) Check your home usage with `df -h ~`.

## Interactive GPU session (first run / debugging)

```bash
srun -p free-gpu --gres=gpu:1 -c 4 --mem=32G -t 02:00:00 --pty /bin/bash -i
module load python/3.10.2
source $ENV_DIR/bin/activate
cd /pub/$USER/Stable-Diffusion

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
python scripts/generate.py --model sd15 --prompt "a lighthouse on a cliff at dusk, oil painting"
python scripts/generate.py --model sdxl --prompt "a red fox in a snowy birch forest, golden hour" --preset balanced
python scripts/parity_check.py --model sd15        # optional: pip install diffusers transformers accelerate
```

## Batch jobs

```bash
cd /pub/$USER/Stable-Diffusion
sbatch hpc3/benchmark.slurm          # tests + toy study + memory ladders + sampler comparisons (several hours)
sbatch hpc3/generate.slurm "a watercolor fox in a snowy forest"
squeue -u $USER                      # job status
tail -f slurm-sd-scratch-bench-<jobid>.out
```

Outputs land in `results/` and `outputs/hpc3/`. Copy them to your laptop with
`rsync -av <ucinetid>@hpc3.rcic.uci.edu:/pub/<ucinetid>/Stable-Diffusion/results/ results/`.
