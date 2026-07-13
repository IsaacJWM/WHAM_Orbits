#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=partition_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --mail-type=END
#SBATCH --mail-user=isaacwrightmuller@gmail.com

conda activate particle_orbits

python Firebird_testing.py
