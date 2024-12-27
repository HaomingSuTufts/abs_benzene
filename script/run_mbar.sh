#!/bin/bash
#SBATCH --job-name=abs_benzene_energy
#SBATCH --output=./log/abs_benzene_mbar.out
#SBATCH --error=./log/abs_benzene_mbar.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=dinglab
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate hfe

python /cluster/tufts/dinglab/hsu02/code/openatom/example/abs_benzene/script/run_mbar.py --phase 'water'