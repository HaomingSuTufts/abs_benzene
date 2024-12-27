# create  input files for FEP simulation

import argparse
import os
import pickle
import openmm as mm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging
from utils import setup_logging, monitor_performance
from openatom.functions import (
    make_psf_from_topology
)
from function import make_abs_alchemy_system
from config import SimulationConfig


class PathManager:
    def __init__(self, molecule_name: str, phase: str):
        self.molecule_name = molecule_name
        self.phase = phase
        self.base_path = Path(__file__).resolve().parents[1]
        self.data_path = self.base_path / 'data'
        self.structure_path = self.base_path / 'structure'
        self.output_path = self.base_path / 'output'

    @property
    def pdb_file(self) -> Path:
        return self.structure_path / f'{self.molecule_name}.pdb'

    @property
    def topology_file(self) -> Path:
        return self.structure_path / f'{self.molecule_name}.prmtop'

    @property
    def solvent_topology_file(self) -> Path:
        return self.structure_path / 'solvent.prmtop'

    @property
    def inpcrd_file(self) -> Path:
        return self.structure_path / f'{self.molecule_name}.inpcrd'

    @property
    def solvent_inpcrd_file(self) -> Path:
        return self.structure_path / 'solvent.inpcrd'

    def lambda_file(self) -> Path:
        return self.output_path / f'{self.phase}_phase' / 'lambdas.pkl'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create input files for FEP simulation')
    parser.add_argument('--molecule_name', type=str,
                        help='Name of the molecule')
    parser.add_argument('--phase', type=str, help='Phase of the molecule')
    return parser.parse_args()


@monitor_performance
def main():
    args = get_args()
    path_manager = PathManager(args.molecule_name, args.phase)
    system_config = SimulationConfig.DEFAULT_SETTINGS

    # print configuration
    logging.info(f'Configuration: {system_config}')
    output_phase_path = path_manager.output_path / f'{args.phase}_phase'
    output_phase_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_phase_path, f'{args.phase}_phase_env_setup')
    logging.info(f'Creating input files for {
                 args.molecule_name} in {args.phase} phase')
    logging.info(f'Path: {path_manager.base_path}')

    envi_prmtop = app.AmberPrmtopFile(str(path_manager.solvent_topology_file))
    envi_system = envi_prmtop.createSystem(
        nonbondedMethod=system_config['nonbondedMethod'],
        nonbondedCutoff=system_config['nonbondedCutoff'],
        constraints=system_config['constraints'],
        switchDistance=system_config['switchDistance'],
    )
    envi_top = envi_prmtop.topology
    envi_coor = app.AmberInpcrdFile(
        str(path_manager.solvent_inpcrd_file)).getPositions()
    envi_coor = np.array(envi_coor.value_in_unit(system_config['lenth_unit']))

    lig_prmtop = app.AmberPrmtopFile(
        str(path_manager.topology_file), envi_top.getPeriodicBoxVectors()
    )
    lig_system = lig_prmtop.createSystem(
        nonbondedMethod=system_config['nonbondedMethod'],
        nonbondedCutoff=system_config['nonbondedCutoff'],
        constraints=system_config['constraints'],
        switchDistance=system_config['switchDistance'],
    )
    lig_top = lig_prmtop.topology
    lig_coor = app.AmberInpcrdFile(
        str(path_manager.inpcrd_file)).getPositions()
    lig_coor = np.array(lig_coor.value_in_unit(unit.nanometer))

    env_xml = XmlSerializer.serialize(envi_system)
    lig_xml = XmlSerializer.serialize(lig_system)

    # save the xml files
    with open(output_phase_path / 'env.xml', 'w') as f:
        f.write(env_xml)
    with open(output_phase_path / 'lig.xml', 'w') as f:
        f.write(lig_xml)
    env = ET.fromstring(env_xml)
    lig = ET.fromstring(lig_xml)

    with open(path_manager.lambda_file(), "wb") as f:
        pickle.dump(SimulationConfig.lambda_list, f)

    for lambdas in SimulationConfig.lambda_list:

        if args.phase.lower() == 'vacuum':
            envi_data = None
            envi_top_data = None
            envi_coor_data = None
        else:
            envi_data = env
            envi_top_data = envi_top
            envi_coor_data = envi_coor

        system_xml, top, coor = make_abs_alchemy_system(
            lig, lig_top, lig_coor, lambdas, envi_data, envi_top_data, envi_coor_data)

        tree = ET.ElementTree(system_xml)
        ET.indent(tree, space="\t", level=0)
        elec, vdw = lambdas[0][0], lambdas[0][1]

        sys_dir = output_phase_path / 'sys'
        sys_dir.mkdir(parents=True, exist_ok=True)

        xml_filename = f"{elec:.2f}_{vdw:.2f}.xml"
        xml_path = sys_dir / xml_filename
        tree.write(
            xml_path,
            xml_declaration=True,
            method="xml",
            encoding="utf-8",
        )

        pdb_path = output_phase_path / 'system.pdb'
        with open(pdb_path, 'w') as pdb_file:
            app.PDBFile.writeFile(
                top, coor * 10, pdb_file, keepIds=True
            )

        topology_path = output_phase_path / 'topology.pkl'
        with open(topology_path, "wb") as file_handle:
            pickle.dump(top, file_handle)

        psf_path = output_phase_path / 'system.psf'
        make_psf_from_topology(top, str(psf_path))


if __name__ == '__main__':
    main()
