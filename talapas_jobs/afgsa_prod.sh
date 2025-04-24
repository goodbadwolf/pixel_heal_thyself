#!/bin/bash
#SBATCH --account=cdux
#SBATCH --job-name=afgsa_prod
#SBATCH --output=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_prod_out.txt
#SBATCH --error=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_prod_err.txt
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu-40gb

cd /gpfs/projects/cdux/mmathai/pixel_heal_thyself
source .venv/bin/activate
uv run python -m pht.train -cn prod trainer.epochs=8 data.patches.patch_size=64
