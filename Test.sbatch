#!/bin/bash
#SBATCH --job-name=thermal_run_test
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=isaacwrightmuller@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate particle_orbits

python Main.py
