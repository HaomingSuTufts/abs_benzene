import argparse
import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import mdtraj
from utils import setup_logging
from config import SimulationConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FEP simulation for molecule")
    parser.add_argument(
        "--phase", choices=["vacuum", "water"], default="water")
    return parser.parse_args()


def get_slurm_array_task_id() -> int:
    try:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        logging.error("Failed to get SLURM_ARRAY_TASK_ID")
        raise


def load_lambdas(phase: str) -> List[List[Tuple[float, float]]]:
    lambdas_path = Path(f"./output/{phase}_phase/lambdas.pkl")
    if not lambdas_path.is_file():
        logging.error(f"Failed to find lambdas file: {lambdas_path}")
        raise FileNotFoundError(f"Lambda file not found: {lambdas_path}")
    with lambdas_path.open("rb") as f:
        return pickle.load(f)


def load_system(phase: str, lambdas: List[Tuple[float, float]]) -> mm.System:
    elec, vdw = lambdas[0][0], lambdas[0][1]

    xml_path = Path(f"./output/{phase}_phase/sys/{elec:.2f}_{vdw:.2f}.xml")
    if not xml_path.is_file():
        logging.error(f"Failed to find system XML file: {xml_path}")
        raise FileNotFoundError(f"System XML file not found: {xml_path}")
    with xml_path.open("r") as f:
        system_xml = f.read()
    return mm.XmlSerializer.deserialize(system_xml)


def add_barostat(system: mm.System, phase: str):
    if phase == "water":
        system.addForce(mm.MonteCarloBarostat(
            1 * unit.atmospheres, 298.15 * unit.kelvin))


def main():
    args = get_args()
    phase = args.phase
    config = SimulationConfig()
    setup_logging(f"./output/{phase}_phase", task_name="compute_energy")
    logging.info(f"Running compute_energy.py for {phase} phase")

    try:
        idx_lambda = get_slurm_array_task_id()
    except KeyError:
        idx_lambda = 1

    try:
        lambdas_list = load_lambdas(phase)
    except FileNotFoundError as e:
        logging.error(e)
        return

    if idx_lambda < 0 or idx_lambda >= len(lambdas_list):
        logging.error(f"idx_lambda ({idx_lambda}) out of range")
        return

    lambdas = lambdas_list[idx_lambda]
    elec, vdw = lambdas[0][0], lambdas[0][1]
    lambdas_str = f"{elec:.2f}_{vdw:.2f}"
    logging.info(f"Running compute_energy.py for {lambdas_str}")

    try:
        system = load_system(phase, lambdas)
    except FileNotFoundError as e:
        logging.error(e)
        return

    add_barostat(system, phase)

    topology_path = Path(f"./output/{phase}_phase/topology.pkl")
    if not topology_path.is_file():
        logging.error(f"The topology file is not found: {topology_path}")
        return

    with topology_path.open("rb") as f:
        topology_openmm = pickle.load(f)

    topology = mdtraj.Topology.from_openmm(topology_openmm)

    pdb_path = Path(f"./output/{phase}_phase/system.pdb")
    if not pdb_path.is_file():
        logging.error(f"The PDB file is not found: {pdb_path}")
        return

    integrator = mm.LangevinMiddleIntegrator(
        config.settings["temperature"],
        config.settings["friction"],
        config.settings["timestep"]
    )
    kbT = config.settings["kbT"]

    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(topology, system, integrator, platform)

    reduced_u = []
    for lambdas in lambdas_list:
        elec, vdw = lambdas[0][0], lambdas[0][1]
        lambdas_traj = f"{elec:.2f}_{vdw:.2f}"
        try:
            traj_path = Path(f"./output/{phase}_phase/traj/{lambdas_traj}.dcd")
            if not traj_path.is_file():
                logging.error(f"Failed to find trajectory file: {traj_path}")
                raise FileNotFoundError(
                    f"Trajectory file not found: {traj_path}")

            traj = mdtraj.load(str(traj_path), top=topology)

            for xyz, unit_cell_vectors in zip(traj.xyz, traj.unitcell_vectors):
                simulation.context.setPositions(xyz)
                simulation.context.setPeriodicBoxVectors(*unit_cell_vectors)
                u = simulation.context.getState(
                    getEnergy=True).getPotentialEnergy() / kbT
                reduced_u.append(u)

        except FileNotFoundError as e:
            logging.error(e)
            return

    reduced_u = np.array(reduced_u)
    reduced_potential_dir = Path(f"./output/{phase}_phase/reduced_potentials")
    reduced_potential_dir.mkdir(parents=True, exist_ok=True)
    reduced_potential_pkl = reduced_potential_dir / f"{lambdas_str}.pkl"

    with reduced_potential_pkl.open("wb") as f:
        pickle.dump(reduced_u, f)

    logging.info(f"Successfully saved reduced potentials to {
                 reduced_potential_pkl}")


if __name__ == "__main__":
    main()
