#!/bin/bash
#SBATCH --account=cdux
#SBATCH --job-name=afgsa_dev
#SBATCH --output=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_dev_out.txt
#SBATCH --error=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_dev_err.txt
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1

cd /gpfs/projects/cdux/mmathai/pixel_heal_thyself
source .venv/bin/activate
uv run python -m pht.train -cn dev
