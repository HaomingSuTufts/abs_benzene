import os
import sys
import logging
import argparse
from typing import Optional, List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from utils import setup_logging


VALID_FORCE_FIELDS = ['MMFF94', 'UFF']
SUPPORTED_FILE_TYPES = ['.csv', '.txt']

def get_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='Convert SMILES to 3D SDF files')
    parser.add_argument('--working_type', type=str, 
                       choices=['single', 'multi'],
                       help='single or multi smiles to input',
                       required=True)
    parser.add_argument('--smiles', type=str,
                       help='SMILES string or path to file with SMILES',
                       required=True)
    parser.add_argument('--output_dir', type=str,
                       help='Output directory path',
                       required=True)
    parser.add_argument('--force_field', type=str,
                       choices=VALID_FORCE_FIELDS,
                       default='MMFF94',
                       help='Force field for MD simulation')
    parser.add_argument('--simulation_time', type=int,
                       default=1000,
                       help='Number of simulation time steps')
    return parser.parse_args()



def validate_inputs(smiles_path: str, output_dir: str) -> None:
    """Validate input parameters and create output directory if needed."""
    if os.path.isfile(smiles_path):
        if not any(smiles_path.endswith(ext) for ext in SUPPORTED_FILE_TYPES):
            raise ValueError(f'Input file must be one of: {", ".join(SUPPORTED_FILE_TYPES)}')
        if not os.path.exists(smiles_path):
            raise FileNotFoundError(f"Input file not found: {smiles_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory not writable: {output_dir}")



def sanitize_smiles_to_filename(smiles: str) -> str:
    """Sanitize SMILES string to use as a filename."""
    return smiles.replace('/', '_').replace('\\', '_').replace(':', '_')



def smiles_to_input(smiles: str, output_path: str,
                         force_field: str = 'MMFF94',
                         simulation_time: int = 1000) -> bool:
    """
    Convert SMILES string to 3D SDF file using RDKit.

    Args:
        smiles (str):
        output_path (str): 
        force_field (str): 
        simulation_time (int): 

    Returns:
        bool: 
    Raises:
        ValueError: 
        RuntimeError: 
    """
    try:

        if not isinstance(smiles, str) or not smiles:
            raise ValueError("Invalid SMILES string")
        
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("Invalid output path")

        if force_field not in ['MMFF94', 'UFF']:
            raise ValueError("Force field must be either 'MMFF94' or 'UFF'")

        if not isinstance(simulation_time, int) or simulation_time <= 0:
            raise ValueError("Simulation time must be a positive integer")
        
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)


        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Failed to parse SMILES: {smiles}")

        filename = sanitize_smiles_to_filename(smiles)
        output_path = os.path.join(output_path, f'{filename}.sdf')
        mol = Chem.AddHs(mol)


        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success == -1:
            raise RuntimeError("Failed to generate 3D coordinates")

      
        if force_field == 'MMFF94':
            success = AllChem.MMFFOptimizeMolecule(mol, maxIters=simulation_time)
        else: 
            success = AllChem.UFFOptimizeMolecule(mol, maxIters=simulation_time)

        if success == -1:
            raise RuntimeError(f"Force field optimization failed using {force_field}")


        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()

        logging.info(f'Successfully processed {smiles} -> {output_path}')
        return True

    except Exception as e:
        logging.error(f"Error processing {smiles}: {str(e)}")
        raise




def process_file(file_path: str, output_dir: str, force_field: str, 
                simulation_time: int) -> None:
    """Process multiple SMILES from a file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if 'smiles' not in df.columns:
            raise ValueError("CSV file must contain 'smiles' column")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
            try:
                output_path = output_dir
                smiles_to_input(row['smiles'], output_path, force_field, simulation_time)
            except Exception as e:
                logging.error(f'Failed processing row {idx}: {e}')
                continue

    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing SMILES"):
                try:
                    smiles = line.strip()
                    output_path = output_dir
                    smiles_to_input(smiles, output_path, force_field, simulation_time)
                except Exception as e:
                    logging.error(f'Failed processing line {idx}: {e}')
                    continue




def main():
    """Main execution function."""
    try:
        args = get_args()
        setup_logging(args.output_dir, task_name='smiles_to_input')
        validate_inputs(args.smiles, args.output_dir)

        if args.working_type == 'single':
            output_path = args.output_dir
            smiles_to_input(args.smiles, output_path, 
                          args.force_field, args.simulation_time)
        else:
            process_file(args.smiles, args.output_dir, 
                        args.force_field, args.simulation_time)
            
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()