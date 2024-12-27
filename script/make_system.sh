#!/bin/bash
#SBATCH --job-name=abs
#SBATCH --output=./log/abs.out
#SBATCH --error=./log/abs.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=dinglab


source ~/.bashrc
conda activate hfe

python /cluster/tufts/dinglab/hsu02/code/openatom/example/abs_benzene/script/make_system.py --molecule_name 'flu' --phase 'water'