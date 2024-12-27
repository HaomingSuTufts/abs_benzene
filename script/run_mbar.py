
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import logging
from utils import setup_logging, monitor_performance
from FastMBAR import FastMBAR
import argparse
from config import SimulationConfig


def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FEP simulation for molecule")
    parser.add_argument("--phase", type=str,
                        help="Phase of the molecule", default="water")
    return parser.parse_args()


def load_lambdas(phase: str) -> List[List[Tuple[float, float]]]:

    lambdas_path = Path(f"./output/{phase}_phase/lambdas.pkl")
    if not lambdas_path.is_file():
        raise FileNotFoundError(f"lambda file not found: {lambdas_path}")

    with lambdas_path.open("rb") as f:
        lambdas_list = pickle.load(f)

    return lambdas_list


def load_reduced_potentials(phase: str, lambdas_list: List[Tuple[float, float]]) -> List[np.ndarray]:
    u_list = []
    for lambdas in lambdas_list:
        elec, vdw = lambdas[0][0], lambdas[0][1]
        lambdas_str = f"{elec:.2f}_{vdw:.2f}"
        u_path = Path(
            f"./output/{phase}_phase/reduced_potentials/{lambdas_str}.pkl")

        if not u_path.is_file():
            raise FileNotFoundError(
                f"reduced potential file not found: {u_path}")

        with u_path.open("rb") as f:
            u = pickle.load(f)

        u_list.append(u)

    return u_list


@monitor_performance
def main():

    setup_logging("./output/", task_name="mbar")
    config = SimulationConfig()

    args = get_args()
    phase = args.phase
    try:
        lambdas_list = load_lambdas(phase)
        logging.info(f"lambdas {phase} phase: {lambdas_list}")
        u_list = load_reduced_potentials(phase, lambdas_list)
        u = np.array(u_list)
        num_confs = np.array([u.shape[1] // u.shape[0]] * u.shape[0]) 
        '''
        expected shape of u: (num_lambdas, nun_confs * num_lambdas)
        '''

        logging.info(f"number of configurations: {num_confs}")
        mbar = FastMBAR(u, num_confs, cuda=True,
                        verbose=True, method="L-BFGS-B")
        kbT = config.settings["kbT"]
        kbT = kbT.value_in_unit(config.settings["report_energy_unit"])
        F = mbar.F * kbT
        F_h = F[0] - F[-1]
        logging.info(f"{phase} phase free energy computed.")
        logging.info(f"{phase} phase free energy difference: {F_h}")
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"{phase} phase free energy computation failed: {e}")


if __name__ == "__main__":
    main()
