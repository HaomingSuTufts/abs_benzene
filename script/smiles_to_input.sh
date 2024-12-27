#!/bin/bash
#SBATCH --job-name=abs
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=dinglab

source ~/.bashrc
conda activate hfe

python /cluster/tufts/dinglab/hsu02/code/openatom/example/abs_benzene/script/smiles_to_input.py --working_type 'single' --smiles 'C[C@@H](c1ccc(c(c1)F)c2ccccc2)C(=O)O' --output_dir '/cluster/tufts/dinglab/hsu02/code/openatom/example/abs_benzene/structure'