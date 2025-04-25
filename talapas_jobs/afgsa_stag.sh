#!/bin/bash
#SBATCH --account=cdux
#SBATCH --job-name=afgsa_stag
#SBATCH --output=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_p64_n200_r0.5/run043/afgsa_stag_out.txt
#SBATCH --error=/gpfs/projects/cdux/mmathai/pixel_heal_thyself/runs/afgsa_p64_n200_r0.5/run043/afgsa_stag_err.txt
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=gpu-40gb

cd /gpfs/projects/cdux/mmathai/pixel_heal_thyself
source .venv/bin/activate
uv run python -m pht.train -cn stag data.resize=0.5 trainer.epochs=20 run_num=43 trainer.curve_order=zorder trainer.use_multiscale_discriminator=false trainer.use_film=true
