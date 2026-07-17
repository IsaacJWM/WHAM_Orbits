#!/bin/bash
#SBATCH --job-name=Orbits_run
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=isaacwrightmuller@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate particle_orbits

python Main.py
