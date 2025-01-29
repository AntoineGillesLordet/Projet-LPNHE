#!/bin/bash
#SBATCH -p rossw
#SBATCH --job-name=2M++xZTF
#SBATCH --nodes=1
#SBATCH --time=6:00:00

source ~/conda-env/setup-environment.sh -i;
conda activate lemaitre;

# srun python make_sample_faster.py
srun -c 128 python make_Tgrid_faster.py
srun python make_cut_faster.py