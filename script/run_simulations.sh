#!/bin/bash
#SBATCH --job-name=abs
#SBATCH --array=0-19
#SBATCH --output=./log/abs_simulation_%A_%a.out
#SBATCH --error=./log/abs_simulation_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=dinglab
#SBATCH --gres=gpu:1



source ~/.bashrc
conda activate hfe

python /cluster/tufts/dinglab/hsu02/code/openatom/example/abs_benzene/script/run_simulations.py --phase 'water' 
